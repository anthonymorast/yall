#ifndef YALL_GRADIENT_DESCENT
#define YALL_GRADIENT_DESCNET

#include <YALL/Models/NeuralNet/Optimizer.hpp>

namespace yall 
{

    class BackPropagation: public Optimizer
    {
        public:
            BackPropagation(double learning_rate) { _learning_rate = learning_rate; }
            virtual void calculate_updates(std::vector<Layer> &layers, double *output, double* target, int output_size);
            virtual void apply_updates(std::vector<Layer> &layers);
        private:
            double _learning_rate;
    };

    void BackPropagation::calculate_updates(std::vector<Layer> &layers, double* output, double* target, int output_size)
    {
        // Notes: https://sudeepraja.github.io/Neural/  + about 10 books

        //TODO: allow user to specify loss function.
        // for now I'm using 0.5*(y_hat - y)^2
        Layer output_layer = layers.back();
        arma::mat delta = output_layer.apply_activation_derivative();
        for(int i = 0; i < output_size; i++)
        {
            // loss derivative * activation derivative
            delta[i] *= (target[i] - output[i]);        
        }
       
        int count = 0; 
        for(auto it = layers.rbegin(); it != layers.rend(); it++)
        {
            Layer &layer = *it;
            arma::mat dedwi = delta*layer.get_input().t();
            arma::mat ldeltas = _learning_rate * (layer.get_weights() % dedwi);
            layer.set_deltas(ldeltas);

            // update delta: delta = (current layer weights)*delta (hadamard) (previous layer activation derivatives)
            // Note: because we are doing 'rebgin() = reverse begin' adding to the iterator gets the previous layer
            arma::mat last_value;
            if(count == (layers.size()-1))     // use network/layer input
            {
                last_value = layer.get_input();
            } 
            else // use previous layer activation derivative
            {
                last_value = (*(it+1)).apply_activation_derivative();
            }
            delta = (layer.get_weights().t() * delta) % last_value;
            count++;
        }
    }

    void BackPropagation::apply_updates(std::vector<Layer> &layers) 
    {
        for(auto it = layers.begin(); it != layers.end(); it++)
        {
            (*it).update_weights();
        }
    }
}


#endif 
