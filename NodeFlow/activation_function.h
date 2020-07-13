#pragma once
#include "eigen3/Eigen/Dense"
#include<bits/stdc++.h>
namespace NodeFlow
{
namespace activation
{

class activation_function
{
public:
    int type;
    activation_function(){}
    //since one node has only one value and the activation always y = f(x) so the gradient could be g = g_old*f'(x)
    virtual void compute_value_gradient(double &input, double &compute_gradient)=0;
    virtual void compute_gradient(double delta){}
    int get_type(){return type;}
    //this tow functions only used in PRelu
    virtual void update(){}
    virtual void mul_param_gradient(double gradient){}
    virtual int get_param_number(){return 0;}
    virtual void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset){return;}
};
//the sigmoid function:1/(1+e^(-x))
class sigmoid:public activation_function
{
public:
    sigmoid(){type=NodeFlow::SIGMOID;}
    void compute_value_gradient(double &input, double &compute_gradient)
    {
        input = 1/(1+exp(-input));
        compute_gradient = compute_gradient*input*(1-input);
        return;
    }
};
//the tanh function:(e^x+e^-x)/(e^x+e^(-x))
class tanh:public activation_function
{
public:
    tanh(){type=NodeFlow::TANH;}
    void compute_value_gradient(double &input, double &compute_gradient)
    {
        if (input<-500)
        {
            input = -1;
        }else if (input>500)
        {
            input = 1;
        }else
        {
            double tmp=input;
            input = (exp(input)-exp(-input))/(exp(input)+exp(-input));
            if (std::isnan(input))
            {
                std::cout<<"nan!"<<std::endl;
                std::cout<<"input:"<<tmp<<std::endl;
            }
        }
        compute_gradient = compute_gradient*(1-input*input);
        return;
    }
};
//the Relu function:max(0,x)
class Relu:public activation_function
{
public:
    Relu(){type=NodeFlow::RELU;}
    void compute_value_gradient(double &input, double &compute_gradient)
    {
        compute_gradient = compute_gradient*(input>0?1:0);
        input = std::max(input,0.0);
        return;
    }
};
//the LeakyRelu function:max(ax,x)
class LeakyRelu:public activation_function
{
private:
    double a_;
public:
    LeakyRelu(double a=0.01):a_(a){type=NodeFlow::LEAKYRELU;}
    void compute_value_gradient(double &input, double &compute_gradient)
    {
        compute_gradient = compute_gradient*(input>0?1:a_);
        input = std::max(a_*input,input);
        return;
    }
};
//the PRelu function:max(ax,x)
//ref:  Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification
//set initial a = 0.25
class PRelu:public activation_function
{
private:
    Eigen::MatrixXd *a_;
    double init_a;
    double delta_a;
    double lr_;
    Eigen::MatrixXd *a_gradient;
    int offset_;
    Eigen::MatrixXd g_;
    double momentum_;
public:
    PRelu(double a=0.25,double momentum=0.9,double lr = 0.01):init_a(a),momentum_(momentum),lr_(lr)
    {type=NodeFlow::PRELU;g_=Eigen::MatrixXd(1,1);g_(0,0)=0.0;}
    void compute_value_gradient(double &input, double &compute_gradient)
    {
       compute_gradient = compute_gradient*(input>0?1:a_->block(offset_,0,1,1)(0,0));
       if (input<0)
       {
           g_(0,0) += input;
       }
        input = std::max(0.0,input)+std::min(a_->block(offset_,0,1,1)(0,0)*input,0.0);
        return;
    }
    void compute_gradient(double delta)
    {
        g_ *= delta;
        a_gradient->block(offset_,0,1,1) += g_;
        g_.setZero();
    }
     void update()
    {
        delta_a = momentum_*delta_a + lr_*a_gradient->block(offset_,0,1,1)(0,0);
        a_->block(offset_,0,1,1)(0,0) -=delta_a;
        a_gradient->block(offset_,0,1,1).setZero();
    }
    void mul_param_gradient(double gradient)
    {
        a_gradient->block(offset_,0,1,1) *= gradient;
    }
    int get_param_number(){return 1;}
    void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset)
    {
        a_ = param;
        a_gradient = gradient;
        offset_ = offset;
        offset += 1;
    }
};
//TODO
//the square function:x**2*0.5
class square:public activation_function
{
public:
    void compute_value_gradient(double &input, double &compute_gradient)
    {

        compute_gradient = compute_gradient*input;
        //std::cout<<"compute_gradient"<<compute_gradient<<std::endl;
        //std::cout<<"input"<<input<<std::endl;
        input *= input;
        return;

    }
    int get_param_number(){return 0;}
};
}

}
