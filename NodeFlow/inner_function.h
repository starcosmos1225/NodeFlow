#pragma once
#include "activation_function.h"
#include "eigen3/Eigen/Dense"
#include<bits/stdc++.h>

namespace NodeFlow
{
class inner_function
{
public:

    //the gradient before activation:the d(y_in)
    std::vector<std::shared_ptr<NodeFlow::activation::activation_function> > activation_;
    //the base innerfunction is activation(w*input+bias)
    inner_function(std::shared_ptr<NodeFlow::activation::activation_function> activation=nullptr)
    {
         if (activation)
            activation_.push_back(activation);
    }
    //the input is the input nodes' value
    // param the w and bias
    //value is the compute value
    //the gradient is the compute's gradient
    virtual void compute_value_gradient(const Eigen::MatrixXd &input,
                                        Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr param,
                                        double &value,
                                        double &activation_gradient,
                                        Eigen::MatrixXd &gradient,
                                        bool compute_gradient=true)

    {
        value = (input.transpose().lazyProduct(param))(0,0);
        activation_gradient = 1.0;
        if (activation_.size()>0)
        {
            std::shared_ptr<NodeFlow::activation::activation_function> prelu=nullptr;
            double tmp=1.0;
            for (auto activation:activation_)
            {
                activation->compute_value_gradient(value,activation_gradient);
                if (activation->get_type()==NodeFlow::PRELU)
                {
                    prelu = activation;
                    tmp = activation_gradient;
                }
            }
            if (prelu)
            {
                prelu->mul_param_gradient(activation_gradient/tmp);
            }
        }
        if (compute_gradient)
        {
            //gradient.setOnes();
            //std::cout<<"compute gradient"<<std::endl;
            //gradient = gradient;
            gradient.array() = activation_gradient*input.array();
            //gradient = activation_gradient*gradient.cwiseProduct(input);
        }

    }
    void set_activation_function(std::shared_ptr<NodeFlow::activation::activation_function> activation)
    {
        activation_.push_back(activation);
    }
   std::vector<std::shared_ptr<NodeFlow::activation::activation_function> >& get_activation_function()
   {
       return activation_;
   }
    virtual double compute_input_gradient(int index, Eigen::MatrixXd &param,int node_number=0)
    {
        //std::cout<<"here"<<std::endl;
        //std::cout<<(*param).rows()<<" "<<(*param).cols()<<std::endl;
        return param(index, 0);
    }
    virtual int get_param_number(int input_number,bool is_cnn)
    {
        int ans;
        if (!is_cnn)
            ans = input_number+1;
        else
            ans = 0;
        for (auto activation:activation_)
        {
            ans += activation->get_param_number();
        }
        return ans;
    }
   virtual void set_param_gradient(Eigen::MatrixXd *param,Eigen::MatrixXd *gradient,
                                   int &offset,
                                   int &param_number,
                                   int input_number,
                                   bool is_cnn=false)
   {
       if (!is_cnn)
       {
            param_number = input_number+1;
            offset += param_number;
       }
       for (auto activation:activation_)
       {
           activation->set_param_gradient(param,gradient,offset);
       }
   }
};
class Max_function:public inner_function
{
private:
    //the max index for input
    int max_index;
public:
    Max_function(std::shared_ptr<NodeFlow::activation::activation_function> activation=nullptr):inner_function(activation){}
    void compute_value_gradient(const Eigen::MatrixXd &input,
                                Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr param,
                                double &value,
                                double &activation_gradient,
                                Eigen::MatrixXd &gradient,
                                bool compute_gradient=true)
    {
         int a;
        value = input.block(1,0,input.rows()-1,1).maxCoeff(&max_index,&a);
         activation_gradient = 1.0;
    }
    double compute_input_gradient(int index, Eigen::MatrixXd &param,int node_number=0)
    {

        return index==max_index?1:0;
    }
    int get_param_number(int input_number,bool is_cnn)
    {
        int ans = 0;
        for (auto activation:activation_)
        {
            ans += activation->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd *param,Eigen::MatrixXd *gradient,
                            int &offset,
                            int &param_number,
                            int input_number,
                            bool is_cnn=false)
       {
           for (auto activation:activation_)
           {
               activation->set_param_gradient(param,gradient,offset);
           }
       }
};
class Mean_function:public inner_function
{
public:
    Mean_function(std::shared_ptr<NodeFlow::activation::activation_function> activation=nullptr):inner_function(activation){}
    void compute_value_gradient(const Eigen::MatrixXd &input,
                                Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr param,
                                double &value,
                                double &activation_gradient,
                                Eigen::MatrixXd &gradient,
                                bool compute_gradient=true)
    {
        double mean=0;
         //std::cout<<"value:"<<max_t<<std::endl;
        activation_gradient = 1.0;
        for (int i=1;i<input.rows();++i)
        {
            mean += input(i,0);
        }

        value = mean/(input.rows()-1);
    }
    double compute_input_gradient(int index, Eigen::MatrixXd &param,int node_number=0)
    {
        return 1.0/(node_number);
    }
    int get_param_number(int input_number,bool is_cnn)
    {
        int ans = 0;
        for (auto activation:activation_)
        {
            ans += activation->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd *param,Eigen::MatrixXd *gradient,
                            int &offset,
                            int &param_number,
                            int input_number,
                            bool is_cnn=false)
       {
           for (auto activation:activation_)
           {
               activation->set_param_gradient(param,gradient,offset);
           }
       }
};
class Add_function:public inner_function
{
public:
    Add_function(std::shared_ptr<NodeFlow::activation::activation_function> activation=nullptr):inner_function(activation){}
    void compute_value_gradient(const Eigen::MatrixXd &input,
                                Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr param,
                                double &value,
                                double &activation_gradient,
                                Eigen::MatrixXd &gradient,
                                bool compute_gradient=true)
    {
        Eigen::MatrixXd tmp(input.rows(),1);
        tmp.setOnes();
        tmp(0,0) = 0.0;
        value = (input.transpose().lazyProduct(tmp))(0,0);
        activation_gradient = 1.0;
        if (activation_.size()>0)
        {
            std::shared_ptr<NodeFlow::activation::activation_function> prelu=nullptr;
            double tmp=1.0;
            for (auto activation:activation_)
            {
                activation->compute_value_gradient(value,activation_gradient);
                if (activation->get_type()==NodeFlow::PRELU)
                {
                    prelu = activation;
                    tmp = activation_gradient;
                }
            }
            if (prelu)
            {
                prelu->mul_param_gradient(activation_gradient/tmp);
            }
        }
        if (compute_gradient)
        {
            gradient.setOnes();
            gradient = activation_gradient*gradient.array()*input.array();
        }
    }
    double compute_input_gradient(int index, Eigen::MatrixXd &param,int node_number=0)
    {
        return 1.0;
    }
    int get_param_number(int input_number,bool is_cnn)
    {
        int ans = 0;
        for (auto activation:activation_)
        {
            ans += activation->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd *param,Eigen::MatrixXd *gradient,
                            int &offset,
                            int &param_number,
                            int input_number,
                            bool is_cnn=false)
       {
           for (auto activation:activation_)
           {
               activation->set_param_gradient(param,gradient,offset);
           }
       }
};
class Minus_function:public inner_function
{
public:
    Minus_function(std::shared_ptr<NodeFlow::activation::activation_function> activation=nullptr):inner_function(activation){}
    void compute_value_gradient(const Eigen::MatrixXd &input,
                                Eigen::DenseBase<Eigen::MatrixXd>::BlockXpr param,
                                double &value,
                                double &activation_gradient,
                                Eigen::MatrixXd &gradient,
                                bool compute_gradient=true)
    {
        int node_number = input.rows()-1;
        Eigen::MatrixXd tmp(input.rows(),1);
        tmp.setOnes();
        tmp(0,0) = 0.0;
        for (int i=node_number/2+1;i<=node_number;i++)
        {
            tmp(i,0) = -1.0;
        }
        value = (input.transpose().lazyProduct(tmp))(0,0);
        //std::cout<<"value:"<<value<<std::endl;
        activation_gradient = 1.0;
        //std::cout<<"activation length:"<<activation_.size()<<std::endl;
        if (activation_.size()>0)
        {
            std::shared_ptr<NodeFlow::activation::activation_function> prelu=nullptr;
            double tmp=1.0;
            for (auto activation:activation_)
            {
                activation->compute_value_gradient(value,activation_gradient);
                if (activation->get_type()==NodeFlow::PRELU)
                {
                    prelu = activation;
                    tmp = activation_gradient;
                }
            }
            if (prelu)
            {
                prelu->mul_param_gradient(activation_gradient/tmp);
            }
        }
        //std::cout<<"value after:"<<value<<std::endl;
        if (compute_gradient)
        {
            gradient.setOnes();
            gradient = activation_gradient*gradient.array()*input.array();
        }
    }
    double compute_input_gradient(int index, Eigen::MatrixXd &param,int node_number=0)
    {
        //std::cout<<"index:"<<index<<"node_number:"<<node_number<<std::endl;
        return index<=(node_number/2)?1.0:-1.0;
    }
    int get_param_number(int input_number,bool is_cnn)
    {
        int ans = 0;
        for (auto activation:activation_)
        {
            ans += activation->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd *param,Eigen::MatrixXd *gradient,
                            int &offset,
                            int &param_number,
                            int input_number,
                            bool is_cnn=false)
       {
           for (auto activation:activation_)
           {
               activation->set_param_gradient(param,gradient,offset);
           }
       }
};
}

