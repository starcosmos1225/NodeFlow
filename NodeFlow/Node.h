#pragma once
#include "inner_function.h"
#include "activation_function.h"
#include "optimizer.h"
#include "eigen3/Eigen/Dense"
#include "util.h"
#include <bits/stdc++.h>

namespace NodeFlow
{
//TODO:
//How to share the param with different node?
//CNN need to share the param but in version 1.0, we cannot share the param.

//version 2.0:we use a share_ptr param to share the param with other node, and the param could create by layer.
class Node:public std::enable_shared_from_this<NodeFlow::Node>
{
private:
    //f_ is the compute function
    std::shared_ptr<inner_function> f_;
    //opt_ is the optimizer function
    std::shared_ptr<optimizer> opt_;
    //input is the node which send data to this node
    //use a map to map node to int which the param[i] belong to,so we can get the delta Wi
    std::vector<std::shared_ptr<Node> > input_;
    //output node is the node which this node send data to
    std::vector<std::shared_ptr<Node> > output_;
    //the node's value
    double value_;
    //the node's bias
    double delta_;
    //memory this for compute delta(input)
    double activation_gradient;
    //g_ is the param's current gradient and gradient is the final gradient.
    //gradient may add many times and when update ,the gradient set  to zeros
    //the number of g is same as param
    Eigen::MatrixXd g_;
    Eigen::MatrixXd* gradient_;
    //param is trainable params such as weights and bias
    //the number of param depend on the input and the inner_function.In the version 1.0 the inner_function is only activation(W*input+bias)
    Eigen::MatrixXd* param_;
    int offset_;
    int param_number_;

public:
    Node(std::shared_ptr<NodeFlow::inner_function> f=nullptr,
         std::shared_ptr<NodeFlow::optimizer> opt=nullptr
         );
    //connect to the neibour node
    //type:0 connect to output node
    //        1 connect to input node
    void connect(std::shared_ptr<Node> node, int type);
    //compute the value by inner_function
    void compute(bool compute_gradient=true,bool debug=false);
    //compute the final gradient by inner_function
    void compute_gradient(bool compute_gradient=true);
    //get the  delta
    double get_delta();
    //set the node's delta
    void set_delta(double delta);
    //add the nodes' delta
    void add_delta(double delta);


    //bp for the delta
    void back_propogation_delta();
    //update the param by the optimizer
    void update(double scale=1.0,bool update_param=true);
    //return the node's value
    double get_value();
    //set the nodes' value
    void set_value(double v);
    void set_activation_function(std::shared_ptr<NodeFlow::activation::activation_function> a);
    int get_param_number(bool is_cnn=false);
    //set the param and gradient addr
    void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset,
                            int cnn_param_number=0,bool cnn=false,bool cnn_inner=false);
    //display param
    void display(int index)
    {
        //std::cout<<param_->block(offset_,0,param_number_,1)(index,0)<<std::endl;
        std::cout<<std::setprecision(16)<<value_<<std::endl;
    }
    void initialize_param(std::shared_ptr<NodeFlow::initializer> init);
};
}
NodeFlow::Node::Node(std::shared_ptr<NodeFlow::inner_function> f,
                     std::shared_ptr<NodeFlow::optimizer> opt):
    f_(f),
    opt_(opt)
{
    offset_=-1;
    value_ = 0;
    delta_ = 0;
    g_.resize(1, 1);
    g_(0, 0) = 0.0;
    param_number_ = 0;
}
//to avoid repeat connection,we only connect the input node.
//This guarantee is not implemented inside the node, but in the operation of a higher layer.
void NodeFlow::Node::connect(std::shared_ptr<NodeFlow::Node> node, int type)
{
    if (type==NodeFlow::OUTNODE)
    {
        output_.push_back(node);
        //add a output node means the delta will compute from that node
        //and delta = sum(node[j].param[i]*node[j].delta)
        //so we must connect the node with its param[i]
    }else if (type==NodeFlow::INNODE)
    {
        int row = g_.rows();
        input_.push_back(node);

        //add a input node means need a new param,so the param's length will increase 1.
        g_.conservativeResize(row+1,1);
        g_(row,0) = 0.0;
    }
}

void NodeFlow::Node::compute(bool compute_gradient,bool debug)
{
    Eigen::MatrixXd input(input_.size()+1,1);
    for (int i=1;i<input.size();++i)
    {
        input(i,0) = input_[i-1]->get_value();
    }
    input(0, 0) = 1.0;
    delta_ = 0.0;
    if (f_)
    {
        f_->compute_value_gradient(input, param_->block(offset_,0,param_number_,1), value_, activation_gradient, g_, compute_gradient);
    }
    else
    {
        //TODO raise error
        throw "inner function is not exist!";
    }

}

void NodeFlow::Node::back_propogation_delta()
{
    double before_innerfunction = delta_*activation_gradient;
    //std::cout<<"d:"<<before_innerfunction<<" a:"<<activation_gradient<<"param number:"<<param_number_<<std::endl;
    //int t;
    //std::cin>>t;
    int input_number = input_.size();
    Eigen::MatrixXd p = param_->block(offset_,0,param_number_,1);
    //std::cout<<"p:"<<p<<std::endl;
    for (int i;i!=input_number;++i)
    {
        input_[i]->add_delta(before_innerfunction*f_->compute_input_gradient(
                                 i+1,
                                 p,
                                 input_number));
    }
}

void NodeFlow::Node::compute_gradient(bool compute_gradient)
{

     back_propogation_delta();
    for (auto a:f_->get_activation_function())
    {
        if (a->get_type()==NodeFlow::PRELU)
        {
            a->compute_gradient(delta_);
        }
    }
    if (compute_gradient)
    {
        gradient_->block(offset_,0,param_number_,1).noalias() += delta_*g_;
        delta_ = 0.0;
    }

}


double NodeFlow::Node::get_delta()
{
    return delta_;
}
void NodeFlow::Node::set_delta(double delta)
{
    delta_ = delta;
}
void NodeFlow::Node::add_delta(double delta)
{
    delta_ += delta;
}

void NodeFlow::Node::update(double scale,bool update_param)
{
    if (opt_&&update_param)
    {
        opt_->update(param_,gradient_,offset_,param_number_,scale);
        //(gradient_->block(offset_,0,param_number_,1)).setZero();
    }
    for (auto a:f_->get_activation_function())
    {
        if (a->get_type()==NodeFlow::PRELU)
        {
            a->update();
        }
    }


}

double NodeFlow::Node::get_value()
{
    return value_;
}
void NodeFlow::Node::set_value(double v)
{
    value_ = v;
}
void NodeFlow::Node::set_activation_function(std::shared_ptr<NodeFlow::activation::activation_function> a)
{
    if (f_)
        f_->set_activation_function(a);
}
int NodeFlow::Node::get_param_number(bool is_cnn)
{
    return f_->get_param_number(input_.size(),is_cnn);
}
void NodeFlow::Node::set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,
                                        int &offset,int cnn_param_number,
                                        bool cnn,bool cnn_inner)
{
    //Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr param,Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr gradient

    if (!cnn)
    {
        param_=param;
        gradient_ = gradient;
        offset_ = offset;
        f_->set_param_gradient(param,gradient,offset,param_number_,input_.size(),cnn);
    }else if (!cnn_inner)
    {
        param_=param;
        gradient_ = gradient;
        offset_ = offset;
        param_number_ = cnn_param_number;
        //f_set_param_gradient(param,gradient,param_,gradient_,input_.size(),offset+cnn_param_number*depth,cnn);
    }else
    {
        f_->set_param_gradient(param,gradient,offset,param_number_,input_.size(),cnn);
    }
}
void NodeFlow::Node::initialize_param(std::shared_ptr<NodeFlow::initializer> init)
{
    if (offset_==-1)
        return;
    (*param_)(offset_,0) = 0;
    for (int i=1;i<param_number_;++i)
    {
        (*param_)(offset_+i,0) = init->init();
    }
}
