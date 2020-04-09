#include <YALL/Models.hpp>
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
	std::shared_ptr<yall::Activation> linear = std::make_shared<yall::LinearActivation>();
	std::shared_ptr<yall::Optimizer> gDesc = std::make_shared<yall::GradientDescent>(0.01);
	yall::NeuralNet nn(2, 1);
	nn.add_layer(3, sigmoid);				// add one hidden layer(s)
	nn.add_layer(5, sigmoid);
	nn.add_layer(1, linear);				// add output layer
	nn.train(inputs, trainy, 4, gDesc, 10);

	double** outputs = nn.predict(inputs, 4);
	for(int i = 0; i < 4; i++)
		cout << outputs[i][0] << endl;

	nn.train(inputs, trainy, 4, gDesc, 10);

	outputs = nn.predict(inputs, 4);
	for(int i = 0; i < 4; i++)
		cout << outputs[i][0] << endl;

	return 0;
}
