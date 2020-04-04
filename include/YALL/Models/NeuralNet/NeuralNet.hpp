#ifndef YALL_NN
#define YALL_NN

#include <YALL/Models/NeuralNet/NNParameters.hpp>
#include <YALL/Models/NeuralNet/Activation.hpp>
#include <YALL/Models/NeuralNet/BackPropagation.hpp>

#include <armadillo>
#include <vector>
#include <string>

namespace yall
{

class NeuralNet
{
	public:
		/* Public methods */
		// TODO: will probably need more parameters here
		NeuralNet(int input_size, int output_size);
		void add_layer(int width, Activation activation, double** weights = 0);
		void train(double** inputs, double** outputs, int training_samples, int batch_size=1);
		void save_weights(std::string filename);
		void load_weights(std::string filename);
	private:
		/* Private Data */
		bool _training_complete = false;		// if we've trained, we can't add more layers
		int _hidden_layers = 0;
		int _input_size = 0;
		int _output_size = 0;
		std::vector<int> _hidden_layer_sizes;   // width of each hidden layer
		std::vector<arma::mat> _weights;		// vector of matrices that to define matrix mult between layers
		double* _biases;						// not implemented
		double* _outputs;						
		std::vector<Activation> _activations;	// one activation per layer
		BackPropagation _b_prop;

		/* Private Methods */
		void forward_prop(arma::mat sample);
		void back_prop(double* targets);
};

}

#endif
