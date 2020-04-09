#ifndef YALL_NN
#define YALL_NN

#include <YALL/Models/NeuralNet/NNParameters.hpp>
#include <YALL/Models/NeuralNet/Activation.hpp>
#include <YALL/Models/NeuralNet/Optimizer.hpp>

#include <armadillo>
#include <vector>
#include <string>
#include <memory>	// smart pointers

namespace yall
{

class NeuralNet
{
	public:
		/* Public methods */
		NeuralNet(int input_size, int output_size);
		void add_layer(int width, std::shared_ptr<Activation> activation, double** weights = 0);
		void train(double** inputs, double** outputs, int training_samples, std::shared_ptr<Optimizer> optmizer, 
				int epochs, int batch_size=1);
		double** predict(double** samples, int number_samples);
		void save_weights(std::string filename);
		void load_weights(std::string filename);
	private:
		/* Private Data */
		bool _training_complete = false;		// if we've trained, we can't add more layers
		int _input_size = 0;
		int _output_size = 0;
		int _training_step_size = 1;			// TODO: implement training step size logic
		
		// _layers and _layer_sizes contain the output layer
		int _layers = 0;
		std::vector<int> _layer_sizes;   		// width of each layer except the input layer
		std::vector<arma::mat> _weights;		// vector of matrices that to define matrix mult between layers
		double* _biases;						// not implemented

		// pointers for object slicing: https://stackoverflow.com/questions/8777724/store-derived-class-objects-in-base-class-variables	
		std::vector<std::shared_ptr<Activation>> _activations;	// one activation per layer
		std::shared_ptr<Optimizer> _optimizer;

		/* Private Methods */
		double* forward_prop(arma::mat sample);
		void optimize(double* targets, double* output);
};

}

#endif
