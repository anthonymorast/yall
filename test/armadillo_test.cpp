#include <armadillo>
#include <iostream>
#include <fstream>

int main()
{
	arma::mat a = arma::randu(2, 2);
	arma::mat b = arma::randu(2, 2);
	b.print(std::cout);
	a.print(std::cout);

	std::ofstream fout("arma_test.dat");
	a.raw_print(fout, "tt");
	fout.close();

	std::ofstream fout2("arma_test_no_head.dat");
	b.raw_print(fout2);
	fout2.close();

	arma::mat c = a * b;
	c.print(std::cout);
	return 0;
}
