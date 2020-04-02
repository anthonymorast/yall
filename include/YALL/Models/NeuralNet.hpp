namespace yall
{

class NeuralNet
{
	public:
		NeuralNet();
		void SayHello();
	private:
		int _hidden_layers = 0;
		int _input_size = 0;
		int _output_size = 0;
		double** _weights;
		double* _biases;
};

}
