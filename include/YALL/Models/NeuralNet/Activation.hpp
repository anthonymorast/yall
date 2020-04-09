#ifndef YALL_ACTIVATION
#define YALL_ACTIVATION

#include <armadillo>

namespace yall
{
	/*! An abstract class for neural network activation functions. */
	class Activation
	{
		public:
			/*! Given a set of neuronal inputs, returns the activation function's 
			 *  output.
			 */
			virtual void apply(arma::mat &input) = 0;
			virtual void apply_derivative(arma::mat &input) = 0;
	};

}

#endif
