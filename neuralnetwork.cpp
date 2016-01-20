//
//  neuralnetwork.cpp
//  neuralnetwork
//
//  Created by Ethan Caballero on 1/13/14
//

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/*Inputs to network can be changed at bottom of code*/

struct Node;
typedef std::vector<Node> Level;/*Row of Nodes*/

typedef std::vector<double> Value;
typedef std::vector<int> Topol;

struct Link /*links between nodes*/
{
	double weight;
	double DerivWeight;
};

typedef std::vector<Link> Links;



/* Node Class and its functions -------------------------------------------------------*/
class Node
{
public:
	Node(int OutCount, int idx);
 
	void setOut(double val){OutValue = val;}
	double getOutValue() const { return OutValue; }
	
	void FeedFwd(const Level& level);
 
	void calcOutGradients(double GoalValue);
	void calcHiddenGradients(const Level& nextLevel);
 
	void updateInWeights(Level& level);
 
private:
	double sumDerivWeights(const Level& nextLevel) const;
	
	static double rand0to1() {return rand()/float(RAND_MAX);}/*selects random value between 0 and 1*/
	static double TransFunc(double in) {
		return std::tanh(in);
	}
	static double TransFuncDer(double in) {
		double x = in;
		return 1.0 - in * in; /*1-tanh^2*/
	}
 
	double OutValue;
	int index;
	Links OutWeights;
	double gradient;
 
	static float alpha;  /*momentum*/
	static float eta;    /*learning rate*/
};

float Node::alpha =  0.40;
float Node::eta   =  0.10;

Node::Node(int OutCount, int idx)
{
	for(int i = 0; i < OutCount; ++i){
		Link con;
		con.DerivWeight = 0;
		con.weight = rand0to1();
		
		OutWeights.push_back(con);
	}
 
	index = idx;
}

void Node::FeedFwd(const Level& prevLevel)
{
	double v = 0.0;
 
	for(int n = 0; n < prevLevel.size(); ++n){
		v += prevLevel[n].getOutValue() * prevLevel[n].OutWeights[index].weight;
	}
 
	setOut(TransFunc(v));
}

void Node::calcOutGradients(double GoalValue)
{
	double delta = GoalValue - OutValue;
	gradient = delta * TransFuncDer(OutValue);
}

double Node::sumDerivWeights(const Level& nextLevel) const
{
	double DerivWeights = 0; /*Sum errors of the nodes*/
 
	for(int i = 0; i < nextLevel.size() - 1; ++i){
		DerivWeights += OutWeights[i].weight * nextLevel[i].gradient;
	}
 
	return DerivWeights;
}

void Node::calcHiddenGradients(const Level& nextLevel)
{
	double DerivWeights = sumDerivWeights(nextLevel);
	gradient = DerivWeights * TransFuncDer(OutValue);
}

void Node::updateInWeights(Level& prevLevel)
{
	for(int i = 0; i < prevLevel.size(); ++i){
		Node& Node = prevLevel[i];
		double oldDerivWeight = Node.OutWeights[index].DerivWeight;
		double newDerivWeight = eta * Node.OutValue * gradient + alpha * oldDerivWeight;
		
		Node.OutWeights[index].DerivWeight = newDerivWeight;
		Node.OutWeights[index].weight     += newDerivWeight;
	}
}
/*------------------------------------------------------------------------------------*/



/* Network Class and its functions ---------------------------------------------------------*/
class Network
{
public:
	Network(const Topol& Topol);
 
	void FeedFwd(const Value& In);
	void backProp(const Value& Goal);
	void getOutput(Value& results) const;
	
private:
	typedef std::vector<Level> Levels;
	Levels levels;
 
	double error;
 
	double DisplayError; /*displays error from goal*/
	double DisplaySmoothingFactor;
};

Network::Network(const Topol& Topol)
{
	int levelCount = Topol.size();
 
	for(int levelNum = 0; levelNum < levelCount; ++levelNum){
		levels.push_back(Level());
		int OutCount = levelNum == levelCount - 1 ? 0 : Topol[levelNum+1];
		/*question mark means 0 if else*/
		
		Level& currentLevel = levels.back();
		
		for(int n = 0; n <= Topol[levelNum]; ++n){
			currentLevel.push_back(Node(OutCount, n));
		}
		
		currentLevel.back().setOut(1.0);
	}
}

void Network::FeedFwd(const Value& InVals) {
 
	for(int i = 0; i < InVals.size(); ++i){
		levels[0][i].setOut(InVals[i]);
	}
 
	for(int levelNum = 1; levelNum < levels.size(); ++levelNum){
		Level& level = levels[levelNum];
		const Level& lastLevel = levels[levelNum - 1];
		
		for(int n = 0; n < level.size() - 1; ++n){
			level[n].FeedFwd(lastLevel);
		}
	}
}

void Network::backProp(const Value& Goal) {
	/*Calc RMS error*/
	Level& OutLevel = levels.back();
 
	error = 0.0;/*initialize*/
 
	for(int i = 0; i < Goal.size(); ++i){
		double delta = Goal[i] - OutLevel[i].getOutValue();
		error += delta * delta;
	}
 
	error = std::sqrt(error / Goal.size());
 
	/*current error*/
	DisplayError = (DisplayError * DisplaySmoothingFactor + error)  / (DisplaySmoothingFactor + 1.0);
	
	/*output gradient*/
	for(int i = 0; i < OutLevel.size() - 1; ++i){
		OutLevel[i].calcOutGradients(Goal[i]);
	}
 
	/*hidden gradients*/
	for(int i = levels.size() - 2; i > 0; --i){
		Level& level = levels[i];
		Level& nextLevel = levels[i+1];
		
		for(int j = 0; j < level.size(); ++j){
			level[j].calcHiddenGradients(nextLevel);
		}
	}
 
	/*Update weights*/
	for(int i = levels.size() - 1; i > 0; i--){
		Level& level     = levels[i];
		Level& prevLevel = levels[i-1];
		
		for(int j = 0; j < level.size() - 1; ++j){
			level[j].updateInWeights(prevLevel);
		}
	}
 
};

void Network::getOutput(Value& results) const
{
	results.clear();
	const Level& OutLevel = levels.back();
	for(int i = 0; i < OutLevel.size() - 1; ++i){
		results.push_back(OutLevel[i].getOutValue());
	}
}
/*------------------------------------------------------------------------------------*/


/*functions that process training values & testing values*/
void train(Network& network, Value&& In, Value&& Goal)
{
	network.FeedFwd(In);
	network.backProp(Goal);
}

void test(Network& network, Value&& In)
{
	network.FeedFwd(In);
 
	Value r;
	network.getOutput(r);
 
	for(double val : In){
		printf("%.4f ", val);
	}
 
	printf(": %.4f\n", r[0]);
}
/*-------------------------------------------------------*/


/*for setting input to topology values, training values, & testing values*/
int main()
{
	Topol Topol = {2, 4, 1};
	Network network(Topol);
 
	for(int i = 0; i < 4999; ++i){
		train(network, {0.0, 0.0}, {0.0});
		train(network, {0.0, 1.0}, {1.0});
		train(network, {1.0, 0.0}, {1.0});
		train(network, {1.0, 1.0}, {0.0});
	}
	
	test(network, {0, 0});
	test(network, {1, 0});
	test(network, {0, 1});
	test(network, {1, 1});
}
/*----------------------------------------------------------------*/
