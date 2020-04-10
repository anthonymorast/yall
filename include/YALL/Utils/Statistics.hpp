#ifndef YALL_UTILS_STATS
#define YALL_UTILS_STATS

#include <math.h>

namespace yall 
{
    class Statistics
    {
        public:
            double mean_squared_error(double* y, double* y_hat, int array_size);
            double sum_squared_error(double* y, double* y_hat, int array_size);
            double r_squared(double* y, double* y_hat, int array_size);
            double total_sum_squares(double* y, int array_size);
            double mean(double* x, int array_size);
            double variance(double* x, int array_size);
            double correlation(double* x, double* y, int array_size);
            double standard_deviation(double* x, int array_size);
    };
}

#endif
