#ifndef INTEGRATE_CUDA_H
#define INTEGRATE_CUDA_H

#include "../config_parser/config_parser.h"

struct IntegrationResult {
    double value;
    double abs_error;
    double rel_error;
    long long compute_time_ms;
};

struct IntegrationTask {
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    int func_id;
};

// main CUDA function
IntegrationResult integrate_rectangle_cuda(
    const IntegrationTask& task,
    const Config& cfg,
    double (*func)(double, double),
    int blocks,
    int threads_per_block
);

// Monte-Carlo for CUDA
IntegrationResult integrate_monte_carlo_cuda(
    const IntegrationTask& task,
    const Config& cfg,
    double (*func)(double, double),
    int blocks,
    int threads_per_block
);

#endif // INTEGRATE_CUDA_H