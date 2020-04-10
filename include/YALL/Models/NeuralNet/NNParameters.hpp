#ifndef YALL_NN_PARAMS
#define YALL_NN_PARAMS

namespace yall
{
    // TODO: if we update this to track layers  so each layer can have a different
    // activation function, this will need to change so the layer specific parameters
    // e.g. activation_function, are distinct.
    struct NNParameters 
    {
        int input_layer_size;
        int output_layer_size;
        int number_hidden_layers;
        int learning_rate;
    };
}

#endif
