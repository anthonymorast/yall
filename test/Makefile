CFLAGS=--std=c++11 -Wno-psabi -larmadillo -lDataTable -lYALL 

all: NeuralNet armadillo_test stats linreg dt mr

armadillo_test:
	g++ armadillo_test.cpp -larmadillo -o armadillo_test

NeuralNet:
	g++ -o nn NeuralNetTest.cpp $(CFLAGS)

stats:
	g++ -o stats statistics_test.cpp $(CFLAGS) 

linreg:
	g++ -o linreg LinearRegression.cpp $(CFLAGS) 


dt:
	g++ -o dt DataTableTest.cpp  $(CFLAGS) 

mr:
	g++ -o mr MultipleRegressionTest.cpp $(CFLAGS)

clean:
	rm -rf *.o stats linreg nn armadillo_test mr dt

