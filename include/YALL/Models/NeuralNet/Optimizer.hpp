#ifndef YALL_BACK_PROP
#define YALL_BACK_PROP

#include <YALL/Models/NeuralNet/Layer.hpp>

#include <armadillo>
#include <vector>
#include <memory>

namespace yall
{
    /*! An abstract class that allows the user to define how the 
     * weights will be optimized
     */
    class Optimizer
    {
        public:
            /*! A function to determine what the weight updates should be, i.e. sets
             *  the values in the _del_w array.
             */
            virtual void calculate_updates(std::vector<Layer> &layers, double loss) = 0;
            /*! A function to apply the _del_w weight updates. 
            */
            virtual void apply_updates(std::vector<Layer> &layers) = 0;
            /*! A function that calculate weight update values and applies them.
            */
            void update_and_apply(std::vector<Layer> &layers, double loss)
            {
                calculate_updates(layers, loss);
                apply_updates(layers);
            }

            /*! A variable used to store the weight updates.
            */
            std::vector<arma::mat> _del_w;
    };
}

#endif
