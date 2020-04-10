#ifndef YALL_MULTI_REG
#define YALL_MULTI_REG

#include <YALL/Utils/DataTable.hpp>
#include <string>
#include <armadillo>

namespace yall 
{
    class MultipleRegression 
    {
        public:
            void train(double** x, double* y, int number_samples, int cols);	// least-squares optimization
            void train(DataTable data);
            double* predict(double** x, int number_samples);
            double* predict(DataTable t);
            double get_alpha();
            const double* get_betas();  // don't want these values changed; alternatively, we could deep copy.
            void print_equation(std::ostream& stream);
        private:
            double* _betas = 0;
            double _alpha = 0;
            int _beta_size = 0;
            std::string* _variables;
            bool _training_complete = false;
    };
}

#endif
