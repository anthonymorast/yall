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

	void LinearRegression::train(DataTable table)
	{
		if(!table.has_response())
		{
			std::cout << "ERROR: data table has no response column." << std::endl;
			return;
		}

		if(table.ncols() > 2)
		{
			std::cout << "ERROR: linear regerssion only fits one explanatory variable.\n"
					  << "       Use multiple regression for more complex models.\n"
					  << "       Data table has more than 2 columns for response and explanatory variables."
					  << std::endl;
			return;
		}

		double** xs = table.get_all_explanatory();
		double* y = table.get_response();
		double* x = xs[0];
		train(table.get_all_explanatory()[0], table.get_response(), table.nrows());
		_training_complete = true;
	}

	double* LinearRegression::predict(double* x, int number_samples)
	{
		if (!_training_complete)
			std::cout << "Getting predictions before training." << std::endl;

		double* outputs = new double[number_samples];	
		for(int i = 0; i < number_samples; i++)
		{
			outputs[i] = _beta*x[i] + _alpha;
		}
		return outputs;
	}

	double LinearRegression::get_alpha()
	{
		return _alpha;
	}

	double LinearRegression::get_beta()
	{
		return _beta;
	}

}
