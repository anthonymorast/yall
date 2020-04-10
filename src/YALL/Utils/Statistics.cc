#include <YALL/Utils/Statistics.hpp>

namespace yall
{

    double Statistics::mean_squared_error(double* y, double* y_hat, int array_size)
    {
        double mse = 0;
        for(int i = 0; i < array_size; i++) 
        {
            mse += (y_hat[i] - y[i])*(y_hat[i] - y[i]);
        }
        mse /= array_size;
        return mse;
    }

    double Statistics::sum_squared_error(double* output, double* actual, int array_size)
    {
        double sse = mean_squared_error(output, actual, array_size);
        return (sse*array_size);
    }

    double Statistics::total_sum_squares(double* y, int array_size)
    {
        double m = mean(y, array_size);
        double ss = 0;
        for(int i = 0; i < array_size; i++) 
        {
            double yy = y[i] - m;
            ss += (yy*yy);
        }
        return ss;
    }

    double Statistics::r_squared(double* output, double* actual, int array_size)
    {
        return (1.0 - (sum_squared_error(output, actual, array_size) - total_sum_squares(output, array_size))); 
    }

    double Statistics::mean(double* x, int array_size)
    {
        double mean = 0;
        for(int i = 0; i < array_size; i++)
        {
            mean += x[i];
        }
        return mean/array_size;
    }

    double Statistics::variance(double* x, int array_size)
    {
        double sum = total_sum_squares(x, array_size);
        return sum/(array_size-1);		// Bessel's Correction (n-1)
    }

    double Statistics::correlation(double* x, double* y, int array_size)
    {
        int n = array_size; 	// to make the formula more readable
        double xy = 0;
        double sum_x = 0;
        double sum_y = 0;
        double x2 = 0;
        double y2 = 0;
        for(int i = 0; i < array_size; i++)
        {
            xy += (x[i]*y[i]);
            sum_x += x[i];
            sum_y += y[i];
            x2 += x[i]*x[i];
            y2 += y[i]*y[i];
        }
        return ((n*xy) - (sum_x*sum_y))/((std::sqrt((n*x2) - (sum_x*sum_x))*std::sqrt((n*y2)-(sum_y*sum_y)))); 
    }

    double Statistics::standard_deviation(double* x, int array_size)
    {
        double std_dev = variance(x, array_size);
        return std::sqrt(std_dev);
    }
}
