#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <CL/cl.h>
#include "../config_parser/config_parser.h"

#define CL_CHECK(call) \
    do { \
        cl_int error = call; \
        if (error != CL_SUCCESS) { \
            std::cerr << "OpenCL error " << error << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

const char* kernel_source = R"(
double f1_dejong(double x, double y) {
    double sum = 0.0;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            double base = 5.0 * (i + 2) + (j + 3);
            double dx = x - 16.0 * j;
            double dy = y - 16.0 * i;
            double dx2 = dx * dx, dy2 = dy * dy;
            double dx4 = dx2 * dx2, dy4 = dy2 * dy2;
            double dx6 = dx4 * dx2, dy6 = dy4 * dy2;
            sum += 1.0 / (base + dx6 + dy6);
        }
    }
    return 1.0 / (0.002 + sum);
}

double f2_ackley(double x, double y) {
    const double a = 20.0, b = 0.2, c = 6.283185307179586;
    double sum_sq = x * x + y * y;
    double exp1_arg = -b * sqrt(0.5 * sum_sq);
    if (exp1_arg < -700.0) exp1_arg = -700.0;
    return -a * exp(exp1_arg) - exp(0.5 * (cos(c * x) + cos(c * y))) + a + exp(1.0);
}

double f3_langermann(double x, double y) {
    const int m = 5;
    const double a1[5] = {1, 2, 1, 1, 5}, a2[5] = {4, 5, 1, 2, 4}, c[5] = {2, 1, 4, 7, 2};
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double dx = x - a1[i], dy = y - a2[i];
        double r2 = dx * dx + dy * dy;
        sum += c[i] * exp(-r2 / 3.141592653589793) * cos(3.141592653589793 * r2);
    }
    return -sum;
}

double get_function(int func_id, double x, double y) {
    switch (func_id) {
        case 1: return f1_dejong(x, y);
        case 2: return f2_ackley(x, y);
        case 3: return f3_langermann(x, y);
        default: return 0.0;
    }
}

__kernel void integral_kernel(__global double* result, double x_start, double x_end,
                               double y_start, double y_end, int steps_x, int steps_y, int func_id) {
    int idx = get_global_id(0);
    int total = steps_x * steps_y;
    if (idx >= total) return;

    int ix = idx % steps_x, iy = idx / steps_x;
    double dx = (x_end - x_start) / steps_x;
    double dy = (y_end - y_start) / steps_y;
    double x = x_start + (ix + 0.5) * dx;
    double y = y_start + (iy + 0.5) * dy;

    result[idx] = get_function(func_id, x, y);
}
)";

double get_exact_value(int func_id) {
    switch (func_id) {
        case 1: return 4545447.652;
        case 2: return 857208.2414;
        case 3: return -1.604646665;
        default: return 0;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " func_id config_file steps" << std::endl;
        return 1;
    }

    int func_id = std::stoi(argv[1]);
    std::string config_file = argv[2];
    int steps = std::stoi(argv[3]);

    Config cfg = parse_config(config_file);
    int total_points = steps * steps;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Get platform and device
    cl_platform_id platform;
    cl_uint num_platforms;
    clGetPlatformIDs(1, &platform, &num_platforms);

    cl_device_id device;
    cl_uint num_devices;
    if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices) != CL_SUCCESS) {
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, &num_devices);
    }

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cerr << "Device: " << device_name << std::endl;
    std::cerr << "Points: " << total_points << std::endl;

    // Create context and queue
    cl_int err;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CL_CHECK(err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    CL_CHECK(err);

    // Build program
    size_t source_len = strlen(kernel_source);
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, &source_len, &err);
    CL_CHECK(err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        std::cerr << "Build error: " << log << std::endl;
        return 1;
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "integral_kernel", &err);
    CL_CHECK(err);

    // Allocate memory
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, total_points * sizeof(double), NULL, &err);
    CL_CHECK(err);

    // Set arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 1, sizeof(double), &cfg.x_start);
    clSetKernelArg(kernel, 2, sizeof(double), &cfg.x_end);
    clSetKernelArg(kernel, 3, sizeof(double), &cfg.y_start);
    clSetKernelArg(kernel, 4, sizeof(double), &cfg.y_end);
    clSetKernelArg(kernel, 5, sizeof(int), &steps);
    clSetKernelArg(kernel, 6, sizeof(int), &steps);
    clSetKernelArg(kernel, 7, sizeof(int), &func_id);

    // Execute
    size_t global_size = total_points;
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL));
    CL_CHECK(clFinish(queue));

    // Read results
    double* h_result = new double[total_points];
    CL_CHECK(clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, total_points * sizeof(double), h_result, 0, NULL, NULL));

    // Sum
    double sum = 0.0;
    for (int i = 0; i < total_points; ++i) sum += h_result[i];

    double dx = (cfg.x_end - cfg.x_start) / steps;
    double dy = (cfg.y_end - cfg.y_start) / steps;
    double result_value = sum * dx * dy;

    // Cleanup
    delete[] h_result;
    clReleaseMemObject(d_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    double exact = get_exact_value(func_id);
    double abs_error = std::abs(result_value - exact);
    double rel_error = abs_error / std::abs(exact);

    std::cout << std::setprecision(10) << result_value << std::endl;
    std::cout << std::setprecision(10) << abs_error << std::endl;
    std::cout << std::setprecision(10) << rel_error << std::endl;
    std::cout << elapsed_ms << std::endl;

    return 0;
}