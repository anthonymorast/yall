#include <YALL/Models.hpp>
#include <YALL/Utils.hpp>
#include <DataTable/DataTable.hpp>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
	double** inputs;
	inputs = new double*[4];
	for(int i = 0; i < 4; i++) 
	{
		inputs[i] = new double[2];
	}

	for(int i = 0; i < 4; i++)
		for(int j = 0; j < 2; j++)
			inputs[i][j] = (i + j);

	double **trainy = new double*[4];
	for(int i = 0; i < 4; i++)
	{
		trainy[i] = new double[1];
		trainy[i][0] = i;
	}

	std::shared_ptr<yall::Activation> sigmoid = std::make_shared<yall::SigmoidActivation>();
	std::shared_ptr<yall::Activation> sigmoid2 = std::make_shared<yall::SigmoidActivation>();
	std::shared_ptr<yall::Activation> linear = std::make_shared<yall::LinearActivation>();
	std::shared_ptr<yall::Optimizer> gDesc = std::make_shared<yall::BackPropagation>(0.1);
/*	yall::NeuralNet nn(2, 1);
	nn.add_layer(3, sigmoid);				// add one hidden layer(s)
	nn.add_layer(5, sigmoid);
	nn.add_layer(1, linear);				// add output layer
	nn.train(inputs, trainy, 4, gDesc, 10);

	double** outputs = nn.predict(inputs, 4);
	for(int i = 0; i < 4; i++)
		cout << outputs[i][0] << endl;

	nn.train(inputs, trainy, 4, gDesc, 1);

	outputs = nn.predict(inputs, 4);
	for(int i = 0; i < 4; i++)
		cout << outputs[i][0] << endl;*/

    cout << "\n\n--------------------IRIS--------------------\n\n";

    datatable::DataTable table("iris.data", "class");
    table.shuffle_rows();
    int train_size = table.nrows() * 0.8;
    int test_size = table.nrows() - train_size;

    datatable::DataTable train = table.select_row_range(0, train_size);
    datatable::DataTable test = table.select_row_range(train_size, table.nrows());

    train.print_shape(cout);
    test.print_shape(cout);
    table.print_shape(cout);

    yall::NeuralNet iris_nn(train.ncols() - 1, 1);
    iris_nn.add_layer(10, linear);
//    iris_nn.add_layer(2, sigmoid2);
    iris_nn.add_layer(1, linear);
    iris_nn.train(train, gDesc, 500);

    double** iris_out = iris_nn.predict(test);
    double* iris_act = test.get_response();
    for(int i = 0; i < test.nrows(); i++)
    {
        cout << "Actual: " << iris_act[i] << "\tPredicted: " << iris_out[i][0] << endl;
    }

    /*---------- Contrived Example: https://hmkcode.com/ai/backpropagation-step-by-step/ */
/*    double** input = new double*[1];
    input[0] = new double[2];
    input[0][0] = 2;
    input[0][1] = 3;

    double** weights = new double*[2];
    weights[0] = new double[2];
    weights[1] = new double[2];
    weights[0][0] = 0.11;
    weights[0][1] = 0.12;
    weights[1][0] = 0.21;
    weights[1][1] = 0.08;

    shared_ptr<yall::Activation> linear = make_shared<yall::LinearActivation>();
	std::shared_ptr<yall::Optimizer> gDesc = std::make_shared<yall::BackPropagation>(0.05);
    yall::NeuralNet nn(2, 1);
    nn.add_layer(2, linear, weights);

    delete[] weights;
    weights = new double*[2];
    weights[0] = new double[1];
    weights[1] = new double[1];
    weights[0][0] = 0.14;
    weights[1][0] = 0.15;
    nn.add_layer(1, linear, weights);

    double** output = new double*[1];
    output[0] = new double[1];
    output[0][0] = 1;

    nn.train(input, output, 1, gDesc, 1, 1);*/

	return 0;
}
