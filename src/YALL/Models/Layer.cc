#include <YALL/Models/NeuralNet/Layer.hpp>

namespace yall
{
    Layer::Layer(int width, int previous_width, std::shared_ptr<Activation> activation, double** weights)
    {
        if(weights != 0)
        {
            // TODO: test the weight loading in add_layer() 
            // flatten the array 
            double *w_vals = new double[previous_width * width];
            for(int i = 0; i < previous_width; i++)
            {
                for(int j = 0; j < width; j++)
                {
                    std::cout << i << ", j=" << j << std::endl;
                    w_vals[(width*i) + j] = weights[i][j];
                }
            }
            // use the supplied weights to initialize the matrix
            _weights = arma::mat(w_vals, width, previous_width);
            delete[] w_vals;
        }
        else 
        {	
            _weights = arma::randu(width, previous_width);
        }

        _activation = activation;
        _deltas = arma::mat(width, previous_width, arma::fill::zeros);
        _width = width;
    }

    int Layer::layer_width()
    {
        return _width;
    }

    arma::mat Layer::get_output()
    {
        return _last_output;
    }

    arma::mat Layer::get_input()
    {
        return _last_input;
    }

    arma::mat Layer::apply_activation_derivative()
    {
        arma::mat ret(_last_activation);
        _activation->apply_derivative(ret);
        return ret;
    }

    arma::mat Layer::feed_forward(arma::mat input)
    {
        arma::mat output = _weights*input;
        _last_activation = arma::mat(output);
        _activation->apply(output);
        _last_input = arma::mat(input);
        _last_output = arma::mat(output);
        return output;    
    }

    arma::mat Layer::get_weights()
    {
        return arma::mat(_weights);
    }

    void Layer::update_weights()
    {
        _weights -= _deltas;
    }

    void Layer::set_deltas(arma::mat deltas)
    {
        _deltas = deltas;
    }

}
