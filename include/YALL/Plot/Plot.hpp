#ifndef YALL_PLOT_PLOT_HPP
#define YALL_PLOT_PLOT_HPP

#include <YALL/Plot/PlotParams.hpp>

namespace yall
{
	class Plot
	{
		public:
			void plot_points(double* x, double* y, PlotParams parameters);
	};
}

#endif
