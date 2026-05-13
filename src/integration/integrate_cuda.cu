#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Function 1. De Jong
__device__ double f1_dejong_gpu(double x, double y) {
    double sum = 0.0;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            double base = 5.0 * (i + 2) + (j + 3);
            double dx = x - 16.0 * j;
            double dy = y - 16.0 * i;
            double dx6 = dx * dx * dx * dx * dx * dx;
            double dy6 = dy * dy * dy * dy * dy * dy;
            sum += 1.0 / (base + dx6 + dy6);
        }
    }
    return 1.0 / (0.002 + sum);
}

// Function 2. Ackley
__device__ double f2_ackley_gpu(double x, double y) {
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2.0 * M_PI;
    
    double sum_sq = x * x + y * y;
    double exp1_arg = -b * sqrt(0.5 * sum_sq);
    if (exp1_arg < -700.0) exp1_arg = -700.0;
    
    double term1 = -a * exp(exp1_arg);
    double term2 = -exp(0.5 * (cos(c * x) + cos(c * y)));
    
    return term1 + term2 + a + exp(1.0);
}

// Function 3. Langermann
__device__ double f3_langermann_gpu(double x, double y) {
    const int m = 5;
    const double a1[5] = {1, 2, 1, 1, 5};
    const double a2[5] = {4, 5, 1, 2, 4};
    const double c[5] = {2, 1, 4, 7, 2};
    
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double dx = x - a1[i];
        double dy = y - a2[i];
        double r2 = dx * dx + dy * dy;
        sum += c[i] * exp(-r2 / M_PI) * cos(M_PI * r2);
    }
    return -sum;
}

__device__ double get_function_gpu(int func_id, double x, double y) {
    switch (func_id) {
        case 1: return f1_dejong_gpu(x, y);
        case 2: return f2_ackley_gpu(x, y);
        case 3: return f3_langermann_gpu(x, y);
        default: return 0.0;
    }
}

// CUDA core
__global__ void integral_kernel(
    double* result,
    double x_start, double x_end, 
    double y_start, double y_end,
    int steps_x, int steps_y,
    int func_id
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = steps_x * steps_y;
    
    if (idx >= total) return;
    
    int ix = idx % steps_x;
    int iy = idx / steps_x;
    
    double dx = (x_end - x_start) / steps_x;
    double dy = (y_end - y_start) / steps_y;
    
    double x = x_start + ix * dx + dx / 2.0;
    double y = y_start + iy * dy + dy / 2.0;
    
    result[idx] = get_function_gpu(func_id, x, y);
}

extern "C" void run_cuda_integral(
    double x_start, double x_end,
    double y_start, double y_end,
    int steps_x, int steps_y,
    int func_id,
    double* result_value,
    double* elapsed_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int total_points = steps_x * steps_y;
    int threads_per_block = 256;
    int blocks = (total_points + threads_per_block - 1) / threads_per_block;
    
    std::cerr << "Total points: " << total_points << std::endl;
    std::cerr << "Blocks: " << blocks << ", Threads/block: " << threads_per_block << std::endl;
    
    double* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, total_points * sizeof(double)));
    
    integral_kernel<<<blocks, threads_per_block>>>(
        d_result, x_start, x_end, y_start, y_end, 
        steps_x, steps_y, func_id
    );
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    double* h_result = new double[total_points];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, total_points * sizeof(double), cudaMemcpyDeviceToHost));
    
    double sum = 0.0;
    for (int i = 0; i < total_points; ++i) {
        sum += h_result[i];
    }
    
    double dx = (x_end - x_start) / steps_x;
    double dy = (y_end - y_start) / steps_y;
    *result_value = sum * dx * dy;
    
    delete[] h_result;
    CUDA_CHECK(cudaFree(d_result));
    
    auto end = std::chrono::high_resolution_clock::now();
    *elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}