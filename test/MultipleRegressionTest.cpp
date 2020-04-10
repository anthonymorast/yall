#include <YALL/Models.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace yall;

int main()
{
    string filename = "multiple_regression.csv";

    DataTable data(filename, "x");
    data.print_shape(cout);
    // accidentally saved rownames 
    data.drop_columns(new int[1] {0}, 1);
    // not sure what was with the last row, ditched it
    data.drop_rows(new int[1] {data.nrows() - 1}, 1);
    data.print_shape(cout);

    MultipleRegression mr;
    mr.train(data);     // results match R's lm() function

    double** p = new double*[4];
    for(int i = 0; i < 4; i++)
        p[i] = new double[3];

    for(int i = 0; i < 4; i++)
    {
        p[i][0] = (i+1)*(i+1);
        p[i][1] = (i+1)*(i+1)*(i+1);
    }

    // epsilon column = random noise
    // these are the values I used in R along with the ones above
    // to test the predict function
    p[0][2] = 0.444;
    p[1][2] = 0.666;
    p[2][2] = 0.222;
    p[3][2] = 0.999;

    double* y_hat = mr.predict(p, 4);
    for(int i = 0; i < 4; i++)
        cout << y_hat[i] << endl;

    mr.print_equation(cout);

    return 0;
}
