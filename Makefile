SHELL=/bin/sh
#CC=g++ -g -w -std=c++11
CC=g++ -g -O2

neuralnetwork : neuralnetwork_99.o
	${CC} -o neuralnetwork neuralnetwork_99.o

neuralnetwork_99.o : neuralnetwork_99.cpp 
	${CC} -c neuralnetwork_99.cpp

clean:
	rm neuralnetwork neuralnetwork_99.o  -rf 
