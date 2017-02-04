SHELL=/bin/sh
CC=g++ -g -w -std=c++11

neuralnetwork : neuralnetwork.o
	${CC} -o neuralnetwork neuralnetwork.o

neuralnetwork.o : neuralnetwork.cpp 
	${CC} -c neuralnetwork.cpp

