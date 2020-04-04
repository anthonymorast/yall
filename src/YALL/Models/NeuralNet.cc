#include <YALL/Models/NeuralNet/NeuralNet.hpp>
#include <iostream>
#include <fstream>

namespace yall
{

NeuralNet::NeuralNet(int input_size, int output_size)
{
	_input_size = input_size;
	_output_size = output_size;
}

void NeuralNet::add_layer(int width, Activation activation, double** weights)
{
	//TODO: check for appropraitely sized weight matrices
	if(_training_complete)
	{
		// TODO: we could just remove the last matrix from the weights vector
		// since this one would be the last hidden -> output.
		std::cout << "Attemtped to add layers after training was complete!" << std::endl;
		return;
	}

	// the rows in the weights matrix are either the input_size or the previous 
	// layer depending on if this is the first layer.
	int first_size = _hidden_layers == 0 ? _input_size : _hidden_layer_sizes.back();
	arma::mat w;
	if(weights != 0)
	{
		// TODO: test the weight loading in add_layer() 
		// flatten the array 
		double *w_vals = new double[first_size * width];
		for(int i = 0; i < first_size; i++)
		{
			for(int j = 0; j < width; j++)
			{
				w_vals[(width*i) + j] = weights[i][j];
			}
		}
		// use the supplied weights to initialize the matrix
		w = arma::mat(w_vals, first_size, width);
		delete[] w_vals;
	}
	else 
	{	
		w = arma::randu(first_size, width);
	}

	_weights.push_back(w);
	_activations.push_back(activation);
	_hidden_layer_sizes.push_back(width);
	_hidden_layers++;
}

void NeuralNet::train(double** inputs, double** outputs, int training_size, int batch_size)
{
	// TODO: add functionality to set weights on final layer and set activation function...
	// TODO: maybe just make the output layer added by the user as is done in Keras...
	// add final layer to the weights; either from input -> output or last hidden -> output
	int prev_size = _hidden_layers == 0 ? _input_size : _hidden_layer_sizes.back();
	arma::mat w = arma::randu(prev_size, _output_size);
	_weights.push_back(w);

	// inputs is an array with each row being one example in the dataset
	// Note: sample_count is updated in the batch loop.
	for(int sample_count = 0; sample_count < training_size; )
	{
		//TODO: is my understanding of batching correct?
		// if the # samples is not evenly divisible by the batch size, update batch size
		batch_size = ((sample_count + batch_size) >= training_size) 
						? (training_size - sample_count) 
						: batch_size;
		for(int batch_count = 0; batch_count < batch_size; batch_count++)
		{
			// crate row vector for sample
			arma::mat sample(inputs[sample_count], 1, _input_size);
			forward_prop(sample);
			sample_count++;
		}
	}

	// outputs is similarly defined except for target outputs (responses)
	
	
	_training_complete = true;
}

void NeuralNet::save_weights(std::string model_name)
{
	// only the weights and architecture need to be saved
	// armadillo supports loading and saving matrices: 
	// http://arma.sourceforge.net/docs.html#save_load_mat
	std::cout << "Saving weights is not implemented." << std::endl;

	model_name += ".yall";
	std::ofstream fout(model_name);
	if(!fout.is_open())
	{
		std::cout << "Error opening file: " << model_name << std::endl;
		return;
	}
		
	// read/write format defined in README	
}

void NeuralNet::load_weights(std::string filename)
{
	std::cout << "Loading weights is ot implemented." << std::endl;
}

/** Private Methods **/
void NeuralNet::forward_prop(arma::mat sample)
{
	// send inputs forward through the array
	std::cout << "You're propagating!" << std::endl;
	for(int i = 0; i < _hidden_layers; i++)
	{
		
	}
}

void NeuralNet::back_prop(double* target)
{
	std::cout << "You're propagating, but in reverse!" << std::endl;
	// calculate loss
	// use backprop class to calculate and apply weight updates
}

}
