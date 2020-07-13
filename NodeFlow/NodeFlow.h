#ifndef NODEFLOW_H_INCLUDED
#define NODEFLOW_H_INCLUDED
#include "eigen3/Eigen/Dense"
#include "util.h"
#include "constant.h"

#include "inner_function.h"
#include "activation_function.h"
#include "optimizer.h"
#include "Node.h"
#include "Layer.h"
#include "Graph.h"
#include "Master.h"
#include "Session.h"
#endif // NODEFLOW_H_INCLUDED
/******************************
 *author:Hu xingyu            *
 *Zhejiang university         *
 ******************************/

/*version 1.0: use a network to realize dense layer, the optimizer is RMSProp with activation sigmoid, tanh, relu, leakyrelu, prelu
  version 2.0: add CNN layer,PoolingLayer. Optimizer add: Adam, GradientDescent, Momentum. test the different combination of optimizer
  version 3.0: compute the gradient and param with matrix(to accelerate the compute speed),realize the dropout
  version 4.0: will realize the save and load function
               save:graph name:layer number:each param's value
               load:name->layer if number == read_number: set param's value to the read_param
*/
