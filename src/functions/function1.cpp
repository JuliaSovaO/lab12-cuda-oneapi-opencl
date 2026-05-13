#include "functions.h"
#include <cmath>

// calc 6th degree using multipliation
static inline double pow6(double t) {
    double t2 = t * t;      // t^2
    double t4 = t2 * t2;    // t^4
    return t4 * t2;         // t^6
}

double f1_dejong(double x, double y) {
    double sum = 0.0;

    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            const double base = 5.0 * (i + 2) + (j + 3);
            const double dx = x - 16.0 * j;
            const double dy = y - 16.0 * i;
            const double denom = base + pow6(dx) + pow6(dy);
            sum += 1.0 / denom;
        }
    }

    const double inside = 0.002 + sum;
    return 1.0 / inside;
}