#ifndef YALL_GRADIENT_DESCENT
#define YALL_GRADIENT_DESCNET

#include <YALL/Models/NeuralNet/Optimizer.hpp>

namespace yall 
{

	class GradientDescent: public Optimizer
	{
		public:
			GradientDescent(double learning_rate) { _learning_rate = learning_rate; }
			virtual void calculate_updates(std::vector<arma::mat> weights, double *output, double* target, 
					std::vector<std::shared_ptr<Activation>> activations);
			virtual void apply_updates(std::vector<arma::mat> &weights);
		private:
			double _learning_rate;
	};

	void GradientDescent::calculate_updates(std::vector<arma::mat> weights, double* output, double* target, 
			std::vector<std::shared_ptr<Activation>> activations)
	{
		arma::mat output_layer_weights = weights.back();
		int output_size = output_layer_weights.n_cols;

		double loss = 0;
		for(int i = 0; i < output_size; i++) 
		{
			loss += (target[i] - output[i]);
		}
		loss *= _learning_rate;

		for(int i = 0; i < weights.size(); i++) 
		{	
			arma::mat w(weights[i].n_rows, weights[i].n_cols);
			w.fill(loss);
			_del_w.push_back(w);
		}

	}

	void GradientDescent::apply_updates(std::vector<arma::mat> &weights) 
	{
		for(int i = 0; i < weights.size(); i++)
		{
			weights[i] += _del_w[i];
		}
	}
}


#endif 
