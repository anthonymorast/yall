#ifndef YALL_NN_LAYER
#define YALL_NN_LAYER

#include <YALL/Models/NeuralNet/Activation.hpp>

#include <armadillo>
#include <memory>

namespace yall
{

    class Layer
    {
        public:
            Layer(int width, int previous_width, std::shared_ptr<Activation> activation, double** weights);
            int layer_width();
            arma::mat get_output();
            arma::mat get_input();
            arma::mat feed_forward(arma::mat input);
            arma::mat apply_activation_derivative();
            void update_weights();     // _weights + _deltas
            void set_deltas(arma::mat deltas);
            arma::mat get_weights();
        private:
            arma::mat _last_output;
            arma::mat _last_input;
            int _width;
            std::shared_ptr<Activation> _activation;
            arma::mat _deltas;
            arma::mat _weights;     // from previous layer to this one...
    };

}

#endif
