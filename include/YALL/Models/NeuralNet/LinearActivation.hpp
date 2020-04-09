#ifndef YALL_LINEAR_ACT
#define YALL_LINEAR_ACT

#include <YALL/Models/NeuralNet/Activation.hpp>

namespace yall 
{
	class LinearActivation: public Activation
	{
		public:
			virtual void apply(arma::mat &input);
			virtual void apply_derivative(arma::mat &input);		
	};

	void LinearActivation::apply(arma::mat &input)
	{
		// following suit with TF and Keras, this function returns the unmodified values
		return;
	}

	void LinearActivation::apply_derivative(arma::mat &input)
	{
		input.ones();
	}
}

#endif
