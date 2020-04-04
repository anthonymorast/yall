#include <YALL/Models.hpp>
#include <iostream>

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

	yall::SigmoidActivation act;
	yall::NeuralNet nn(2, 1);
	nn.add_layer(10, act);
	nn.train(inputs, trainy, 4);

	return 0;
}
