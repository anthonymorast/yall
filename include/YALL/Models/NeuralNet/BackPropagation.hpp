#ifndef YALL_BACK_PROP
#define YALL_BACK_PROP

#include <YALL/Models/NeuralNet/NNParameters.hpp>

namespace yall
{
	/*! An abstract class that allows the user to define how the 
	 * loss will be back propagated to update the weights, e.g.
	 * gradient descent. 
	 */
	class BackPropagation
	{
		public:
			virtual ~BackPropagation();
			/*! A function to determine what the weight updates should be, i.e. sets
			 *  the values in the _del_w array.
			 */
			virtual void calculate_updates(double** weights, double *outputs, double* targets, 
											NNParameters params);
			/*! A function to apply the _del_w weight updates. 
			 */
			virtual void apply_updates(double **weights);
			/*! A function that calculate weight update values and applies them.
			 */
			void update_and_apply(double** weights, double* outputs, double* targets,
								  NNParameters params)
			{
				calculate_updates(weights, outputs, targets, params);
				apply_updates(weights);
			}
		private:
			double** _del_w;
	};

	BackPropagation::~BackPropagation()
	{
		// vtables...
	}

	void BackPropagation::apply_updates(double **weights) {}
	void BackPropagation::calculate_updates(double **weights, double *outputs, double *targets, NNParameters params) {}
}

#endif
