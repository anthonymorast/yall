#include <YALL/Models.hpp>
#include <YALL/Plot.hpp>
#include <YALL/Utils.hpp>

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string>
using namespace std;

void fx(double *x, int n)
{
	for(int i = 0; i < n; i++) 
	{
		x[i] = i;
	}
}

void fy(double *x, double *y, int n) 
{
	for(int i = 0; i < n; i++) 
	{
		y[i] = x[i]*x[i];
	}
}

int main()
{
	srand(time(NULL));
	yall::LinearRegression linreg;

	int n = 10;
	double *x = new double[n];
	double *y = new double[n];

	fx(x, n);
	fy(x, y, n);

	double** data = new double*[n];
	for(int i = 0; i < n; i++)
	{
		data[i] = new double[2];
		data[i][0] = x[i];
		data[i][1] = y[i];
	}
	string headers[2] = { "x", "y" };
	yall::DataTable dt(headers, 1, data, n, 2, true);

	cout << "get all" << endl;
	double** d = dt.get_all_explanatory();
	cout << "lreg here" << endl;

	dt.print_headers(cout);
	dt.print_column(cout, 0);
	dt.print_column(cout, 1);
	cout << headers[0] << " " << headers[1] << endl;

	linreg.train(x, y, n);
	cout << linreg.get_beta() << "x + " << linreg.get_alpha() << endl;

	yall::LinearRegression lr;
	lr.train(dt);
	cout << lr.get_beta() << "x + " << lr.get_alpha() << endl;

	return 0;
}
