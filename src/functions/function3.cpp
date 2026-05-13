#include "functions.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double f3_langermann(double x, double y) {
    const int m = 5;
    const double a1[5] = {1, 2, 1, 1, 5};
    const double a2[5] = {4, 5, 1, 2, 4};
    const double c[5] = {2, 1, 4, 7, 2};

    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        double dx = x - a1[i];
        double dy = y - a2[i];
        double r2 = dx * dx + dy * dy;
        sum += c[i] * std::exp(-r2 / M_PI) * std::cos(M_PI * r2);
    }

    return -sum;
}