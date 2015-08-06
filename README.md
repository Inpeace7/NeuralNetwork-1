Here is a neural network in C++ that can learn a function/classifier to get from input to output.  
It uses some C++11 extensions, so I compiled it using:
'''
“clang++ -std=c++11 -stdlib=libc++ neuralnetwork.cpp -o neuralnetwork”
''' 
in the terminal. The input for training values and testing values is currently set to xor and can be changed at the bottom of the code.
