#ifndef YALL_SIG_ACT
#define YALL_SIG_ACT

#include <YALL/Models/NeuralNet/Activation.hpp>
#include <math.h>	// exp

namespace yall 
{
	class SigmoidActivation : public Activation
	{
		public:
			double* apply(double* input, int input_size);
	};

	double* SigmoidActivation::apply(double* input, int input_size)
	{
		double* output = new double[input_size];
		for(int i = 0; i < input_size; i++) 
		{
			output[i] = (1/(1 + exp(input[i])));
		}
		return output;
	}
}

#endif
