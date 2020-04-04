#ifndef YALL_ACTIVATION
#define YALL_ACTIVATION

namespace yall
{
	// Note: this used to be a pure abstract class but I was unable to store a reference
	// in the NeuralNetwork.hpp class. Lack of C++ knowledge or impossible? Thusly, I 
	// decided to just make a useless base class

	/*! An abstract class for neural network activation functions. */
	class Activation
	{
		public:
			virtual ~Activation();
			/*! Given a set of neuronal inputs, returns the activation function's 
			 *  output.
			 */
			virtual double* apply(double* inputs, int input_size);
	};

	Activation::~Activation() 
	{
		// vtables...
	}

	double* Activation::apply(double* inputs, int input_size) { return 0; }
}

#endif
