#pragma once
#include "eigen3/Eigen/Dense"
#include "util.h"
namespace NodeFlow
{


const std::string NONENAME ="None";
const int OUTNODE=1;
const int INNODE=2;
const double LAYER_SCALE = 0.9;
//inner function
const int NORMAL_FUNCTION=0;
const int MAX_FUNCTION=1;
const int MEAN_FUNCTION=2;
const int ADD_FUNCTION=3;
const int MINUS_FUNCTION=4;

//activate function
const int NONE=0;
const int SIGMOID=1;
const int TANH=2;
const int RELU=3;
const int LEAKYRELU=4;
const int PRELU=5;
const int SQUARE=6;
//optimizer
const int RMSPROP=1;
const int GRADIENTDESCENT=2;
const int SGD = 3;
const int MOMENTUM= 4;
const int ADAM= 5;
const int SGD_MULTIPLE = 6;
//class
class Master;
class Graph;
class Layer;
class Node;
class inner_function;
class optimizer;
//max asyn to 10
std::unordered_map<int, Eigen::MatrixXd > RMS_R_MAP[10];
std::unordered_map<int, Eigen::MatrixXd > ADAM_V_MAP[10];
std::unordered_map<int, Eigen::MatrixXd > ADAM_M_MAP[10];
std::unordered_map<int, Eigen::MatrixXd > MOMENTUM_MAP[10];
namespace activation
{
class activation_function;
class sigmoid;
class square;
}
NodeFlow::Constant nf_constant(0,1);
int Adam_iteration;
}
