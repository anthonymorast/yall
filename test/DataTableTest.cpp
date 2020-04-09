#include <YALL/Utils.hpp>

#include <iostream>
using namespace std;

int main()
{
	yall::DataTable dt("x_to_x_squared.csv", "x2", true);
	cout << dt << endl;
	dt.print_headers(cout);
	dt.print_column(cout, 0);
	dt.print_row(cout, 10);

	dt.to_file("same_but_dots.csv", '*');
}
