CFLAGS=-lYALL --std=c++11 -Wno-psabi

all: NeuralNet armadillo_test stats linreg dt

armadillo_test:
	g++ armadillo_test.cpp -larmadillo -o armadillo_test

NeuralNet:
	g++ $(CFLAGS) -o nn NeuralNetTest.cpp

stats:
	g++ $(CFLAGS) -o stats statistics_test.cpp

linreg:
	g++ $(CFLAGS) -o linreg LinearRegression.cpp

dt:
	g++ $(CFLAGS) -o dt DataTableTest.cpp

clean:
	rm -rf *.o stats linreg nn armadillo_test

