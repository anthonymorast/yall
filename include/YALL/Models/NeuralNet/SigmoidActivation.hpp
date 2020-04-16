#ifndef YALL_SIG_ACT
#define YALL_SIG_ACT

#include <YALL/Models/NeuralNet/Activation.hpp>
#include <math.h>	// exp

namespace yall 
{
    class SigmoidActivation : public Activation
    {
        public:
            virtual void apply(arma::mat &input);
            virtual void apply_derivative(arma::mat &input);
        private:
            double logistic_sigmoid(double input);
    };

    void SigmoidActivation::apply(arma::mat &input)
    {
        for(arma::mat::iterator it = input.begin(); it != input.end(); it++)
        {
            (*it) = logistic_sigmoid(*it);
        }	
    }

    void SigmoidActivation::apply_derivative(arma::mat &input)
    {
        std::cout << "before sig: " << std::endl;
        input.print(std::cout);
        for(auto it = input.begin(); it != input.end(); it++)
        {
            (*it) = logistic_sigmoid((*it)) * (1 - logistic_sigmoid((*it)));
        }
        std::cout << "apply sig: " << std::endl;
        input.print(std::cout);
    }

    double SigmoidActivation::logistic_sigmoid(double input)
    {
        return (1/(1+exp(input)));
    }
}

#endif
