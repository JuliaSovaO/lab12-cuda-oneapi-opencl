#include "functions.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double f2_ackley(double x, double y) {
    const double a = 20.0;
    const double b = 0.2;
    const double c = 2.0 * M_PI;

    double sum_sq = x * x + y * y;

    double exp1_arg = -b * std::sqrt(0.5 * sum_sq);
    if (exp1_arg < -700.0) exp1_arg = -700.0;

    double term1 = -a * std::exp(exp1_arg);
    double term2 = -std::exp(0.5 * (std::cos(c * x) + std::cos(c * y)));

    return term1 + term2 + a + std::exp(1.0);
}