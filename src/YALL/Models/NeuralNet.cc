#include <YALL/Models/NeuralNet.hpp>

#include <iostream>

namespace yall
{

NeuralNet::NeuralNet()
{
	std::cout << "You've created a neural network, congratulation!" << std::endl;
}

void NeuralNet::SayHello() 
{
	std::cout << "HELLO!" << std::endl;
}

}
