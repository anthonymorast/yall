#ifndef YALL_MODELS_LOG_REGRESSION
#define YALL_MODELS_LOG_REGRESSION

#include <YALL/Utils/DataTable.hpp>
#include <YALL/Utils/Statistics.hpp>

namespace yall
{

    class LogisticRegression
    {
        public:
            void train(double** x, double* y, int ncols, int nrows);
            void train(DataTable t);
            double* predict(double** x, int nrows);
            double* predict(DataTable t);
            void print_equation(std::ostream& stream);
        private:
            bool _training_complete = false;
            int _number_predictors = 0;
    };

}

#endif
