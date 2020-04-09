#include <YALL/Utils.hpp>

#include <iostream>
using namespace std;

int main()
{
	yall::Statistics stats;
	double *x = new double[10];
	double *y = new double[10];
	for(int i = 0; i < 10; i++)
	{
		x[i] = i;
		y[i] = (i*i);
	}

	cout << stats.correlation(x, y, 10) << endl;
	cout << stats.mean(x, 10) << endl;
	cout << stats.variance(x, 10) << endl;
	cout << stats.standard_deviation(x, 10) << endl;
}
