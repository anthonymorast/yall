#include <YALL/Models/LinearRegression.hpp>

#include <iostream>

namespace yall
{
    void LinearRegression::train(double* x, double* y, int number_samples)
    {
        _beta = (_stats.correlation(x, y, number_samples) * _stats.standard_deviation(y, number_samples));
        _beta /= _stats.standard_deviation(x, number_samples);
        _alpha = _stats.mean(y, number_samples) - (_beta * _stats.mean(x, number_samples));		
        _training_complete = true;
    }

    void LinearRegression::train(DataTable table, std::string variable)
    {
        if(!table.has_response())
        {
            std::cout << "ERROR: data table has no response column." << std::endl;
            return;
        }

        // TODO: test passing in a table with more than 2 columns + variable name
        if(table.ncols() > 2 && variable == "")
        {
            std::cout 
                << "ERROR: the data table has more than 2 columns or the response variable wasn't provided.\n"
                << "       Linear regerssion only fits one explanatory variable.\n"
                << "       Use multiple regression for more complex models."
                << std::endl;
            return;
        }

        double* x;
        if(variable != "")
            x = table.get_column(variable);
        else
            x = table.get_column(table.response_column_number() == 0 ? 1 : 0);

        train(x, table.get_response(), table.nrows());
        _training_complete = true;
    }

    double* LinearRegression::predict(double* x, int number_samples)
    {
        if (!_training_complete)
            std::cout << "WARNING: using linear model before training." << std::endl;

        double* outputs = new double[number_samples];	
        for(int i = 0; i < number_samples; i++)
        {
            outputs[i] = _beta*x[i] + _alpha;
        }
        return outputs;
    }

    double* LinearRegression::predict(DataTable t, std::string variable)
    {
        double* x;
        if(variable != "")
            x = t.get_column(variable);
        else 
            x = t.get_column(t.response_column_number() == 0 ? 1 : 0);
        return predict(x, t.nrows());
    }

    double LinearRegression::get_alpha()
    {
        return _alpha;
    }

    double LinearRegression::get_beta()
    {
        return _beta;
    }

    void LinearRegression::print_equation(std::ostream& stream)
    {
        stream << "beta=" << _beta << "; alpha=" << _alpha << std::endl;
        stream << "y = " << _beta << "x + " << _alpha << std::endl;
    }

}
