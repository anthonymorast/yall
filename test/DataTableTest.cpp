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

	cout << dt.nrows() << ", " << dt.ncols() << endl;
	int* shape = dt.shape();
	cout << shape[0] << ", " << shape[1] << endl;
	dt.print_shape(cout);

	double* col = dt.get_column(0);

	dt.to_file("same_but_dots.csv", '*');

    double** data = new double*[5];
    for(int i = 0; i < 5; i++)
        data[i] = new double[5];

    for(int i = 0; i < 5; i++)
        for(int j = 0; j < 5; j++)
            data[i][j] = (i+j);

    yall::DataTable shuffle_test(data, 5, 5);
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
            cout << data[i][j];
        cout << endl;
    }
    cout << endl << endl;
    shuffle_test.print(cout);
    shuffle_test.shuffle_rows();
    cout << endl << endl;
    for(int i = 0; i < 5; i++)
    {
        for(int j = 0; j < 5; j++)
            cout << data[i][j];
        cout << endl;
    }
    cout << endl << endl;
    shuffle_test.print(cout);
}
