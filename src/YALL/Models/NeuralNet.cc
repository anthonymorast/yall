#include <YALL/Models/NeuralNet/NeuralNet.hpp>

namespace yall
{
    NeuralNet::NeuralNet(int input_size, int output_size)
    {
        _input_size = input_size;
        _output_size = output_size;
        arma::arma_rng::set_seed_random();
    }

    void NeuralNet::add_layer(int width, std::shared_ptr<Activation> activation, double** weights)
    {
        //TODO: check for appropraitely sized weight matrices
        if(_training_complete)
        {
            // since this one would be the last hidden -> output.
            std::cout << "Attemtped to add layers after training was complete!" << std::endl;
            return;
        }

        int prev_size = (_layer_count == 0) ? _input_size : _layers.back().layer_width();
        Layer l(width, prev_size, activation, weights);
        _layers.push_back(l);
        _layer_count++;
    }

    void NeuralNet::train(DataTable table, std::shared_ptr<Optimizer> optimizer, int epochs, int batch_size, double** multi_response)
    {
        double** response;
        if(multi_response != 0)     // e.g one-hot
        {
            response = multi_response;
        }
        else                        // single output neuron
        {
            response = new double*[table.nrows()];
            for(int i = 0; i < table.nrows(); i++)
            {
                response[i] = new double[1];
                response[i][0] = table.get_response()[i];
            }
        }

        train(table.get_all_explanatory(), response, table.nrows(), optimizer, epochs, batch_size);
        delete[] response;
    }

    void NeuralNet::train(double** inputs, double** targets, int training_size, std::shared_ptr<Optimizer> optimizer, 
            int epochs, int batch_size)
    {
        if(!_layer_count)
        {
            std::cout << "The Neural Network doesn't have any layers!" << std::endl;
            return;
        }

        _optimizer = optimizer;
        DataTable dummy(inputs, training_size, _input_size);
        for(int e_count = 0; e_count < epochs; e_count++)
        {
            // WATCH: if we change from storing pointers in DataTable need to update this
            dummy.shuffle_rows();   // should suffle **inputs since we store the pointer.
            for(int sample_count = 0; sample_count < training_size; /*None*/)
            {
                // if the # samples is not evenly divisible by the batch size, update batch size
                batch_size = ((sample_count + batch_size) >= training_size) 
                    ? (training_size - sample_count) 
                    : batch_size;
                double batch_loss;
                for(int batch_count = 0; batch_count < batch_size; batch_count++)
                {
                    arma::mat sample(inputs[sample_count], _input_size, 1);
                    double* output = forward_prop(sample);
                    double* target = targets[sample_count];
                    for(int i = 0; i < _output_size; i++)
                    {
                        batch_loss += (output[i] - target[i]);
                    }
                    sample_count++;
                }
                batch_loss /= batch_size;
                optimize(batch_loss);
            }	
        }

        _training_complete = true;
    }

    double** NeuralNet::predict(DataTable table)
    {
        return predict(table.get_all_explanatory(), table.nrows());
    }

    double** NeuralNet::predict(double** inputs, int number_samples)
    {
        double** outputs = new double*[number_samples];
        for(int i = 0; i < number_samples; i++)
        {
            arma::mat sample(inputs[i], _input_size, 1);
            outputs[i] = forward_prop(sample);
        }

        return outputs;
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
        std::cout << "Loading weights is not implemented." << std::endl;
    }

    /** Private Methods **/
    double* NeuralNet::forward_prop(arma::mat sample)
    {
        double* outputs = new double[_output_size];
        arma::mat last = sample;
        for(auto it = _layers.begin(); it != _layers.end(); it++)
        {
            last = (*it).feed_forward(last);
        }

        // last should be a vector
        int count = 0;
        for(arma::mat::iterator it = last.begin(); it != last.end(); it++)
        {
            outputs[count] = (*it);
            count++;
        }

        return outputs;
    }

    void NeuralNet::optimize(double loss)
    {
        // rely on the BackPropagation class implementations to do the heavy lifting
        _optimizer->update_and_apply(_layers, loss); 
    }

}
