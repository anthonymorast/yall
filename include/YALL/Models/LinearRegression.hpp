#ifndef YALL_LIN_REG
#define YALL_LIN_REG

#include <YALL/Utils/Statistics.hpp>
#include <YALL/Utils/DataTable.hpp>
#include <string>

namespace yall
{
    class LinearRegression 
    {
        public:
            void train(double* x, double* y, int number_samples);	// least-squares optimization
            void train(DataTable data, std::string variable="");
            double* predict(double* x, int number_samples);
            double* predict(DataTable t, std::string variable);
            double get_alpha();
            double get_beta();
            void print_equation(std::ostream& stream);
        private:
            double _beta = 0;
            double _alpha = 0;
            bool _training_complete = false;
            Statistics _stats;
    };
}

#endif
