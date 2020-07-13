#pragma once
#include "eigen3/Eigen/Dense"
#include "eigen3/unsupported/Eigen/MatrixFunctions"
#include <bits/stdc++.h>
//#include "type.h"
namespace NodeFlow
{


//optimizer is a virtual class, it must has child class that implement the optimization throught update function
//extern std::unordered_map<int, Eigen::MatrixXd > RMS_R_MAP;
//extern std::unordered_map<int, Eigen::MatrixXd > MOMENTUM_MAP;
//extern int NodeFlow::Adam_iteration;
class optimizer
{

public:
    optimizer(){}
    //update the param with the gradient
    virtual void update(Eigen::MatrixXd* param, Eigen::MatrixXd* gradient,int offset,int length,double scale=1.0,int asyn_index=0)=0;
    //the set_option could not be called;
    virtual void set_option(double lr=0.01, double prop=0.9, double epsilon=1e-10){
    };
};
class RMSProp:public optimizer
{
private:
    double lr_;
    double prop_;
    double epsilon_;
public:
    RMSProp(double lr=0.01, double prop=0.9, double epsilon=1e-10):lr_(lr),prop_(prop),epsilon_(epsilon){}
     /*r = prop*r+(1-prop)*g^2
     *delta = lr/sqrt(r+epsilon)*g
     *w -= delta
     */

    void update(Eigen::MatrixXd* param, Eigen::MatrixXd* gradient,int offset,int length,double scale=1.0,int asyn_index=0)
    {
        if (NodeFlow::RMS_R_MAP[asyn_index].find(offset)==NodeFlow::RMS_R_MAP[asyn_index].end())
        {
            NodeFlow::RMS_R_MAP[asyn_index][offset] = (1-prop_)*gradient->block(offset,0,length,1).cwiseProduct(gradient->block(offset,0,length,1));
        }else
        {
            NodeFlow::RMS_R_MAP[asyn_index][offset]  = prop_*NodeFlow::RMS_R_MAP[asyn_index][offset] +
                    (1-prop_)*gradient->block(offset,0,length,1).cwiseProduct(gradient->block(offset,0,length,1));
        }
        Eigen::MatrixXd epsilon(length,1);
        //if (NodeFlow::RMS_R_MAP[asyn_index][offset](0,0)>1)
            //std::cout<<"length"<<length<<" g:"<<NodeFlow::RMS_R_MAP[asyn_index][offset]<<std::endl;
        epsilon.setConstant(epsilon_);
        //if (NodeFlow::RMS_R_MAP[asyn_index][offset](0,0)>1)
            //std::cout<<std::setprecision(16)<<"delta:"<<lr_*(gradient->block(offset,0,length,1)).array()/(((NodeFlow::RMS_R_MAP[asyn_index][offset]).array()+epsilon.array()).sqrt())<<std::endl;
        param->block(offset,0,length,1) = (param->block(offset,0,length,1)).array() -
                lr_*scale/(((NodeFlow::RMS_R_MAP[asyn_index][offset]+epsilon).array()).sqrt()) *(gradient->block(offset,0,length,1)).array();
    }
    void set_option(double lr=0.01, double prop=0.9, double epsilon=1e-10)
    {
        lr_=lr;prop_=prop;epsilon_=epsilon;
    }
};
class Adam:public optimizer
{
private:
    double lr_;
    double beta_1_;
    double beta_2_;
    double epsilon_;
public:
    Adam(double lr=0.001, double beta_1=0.9,double beta_2=0.999, double epsilon=1e-7):lr_(lr),beta_1_(beta_1),beta_2_(beta_2),epsilon_(epsilon)
    {
    }
     /*v = beta_2*v+(1-beta_2)*g^2
      *m =beta_1*m+(1-beta_1)*g
      *delta = lr*m/(sqrt(v)+epsilon)
      *w -= delta
     */

    void update(Eigen::MatrixXd* param, Eigen::MatrixXd* gradient,int offset,int length,double scale=1.0,int asyn_index=0)
    {
        if (NodeFlow::ADAM_V_MAP[asyn_index].find(offset)==NodeFlow::ADAM_V_MAP[asyn_index].end())
        {
            NodeFlow::ADAM_V_MAP[asyn_index][offset] = (1-beta_2_)*gradient->block(offset,0,length,1).cwiseProduct(gradient->block(offset,0,length,1));
        }else
        {
            NodeFlow::ADAM_V_MAP[asyn_index][offset]  = beta_2_*NodeFlow::ADAM_V_MAP[asyn_index][offset] +
                    (1-beta_2_)*gradient->block(offset,0,length,1).cwiseProduct(gradient->block(offset,0,length,1));
        }
        if (NodeFlow::ADAM_M_MAP[asyn_index].find(offset)==NodeFlow::ADAM_M_MAP[asyn_index].end())
        {
            NodeFlow::ADAM_M_MAP[asyn_index][offset] = (1-beta_1_)*gradient->block(offset,0,length,1);
        }else
        {
            NodeFlow::ADAM_M_MAP[asyn_index][offset]  = beta_1_*NodeFlow::ADAM_M_MAP[asyn_index][offset] + (1-beta_1_)*gradient->block(offset,0,length,1);
        }
        //No modify:ref tensorflow
        double lr_t;
        lr_t = lr_*sqrt(1-pow(beta_2_,Adam_iteration*1.0))/(1-pow(beta_1_,Adam_iteration*1.0));

        Eigen::MatrixXd epsilon(length,1);
        epsilon.setConstant(epsilon_);
        param->block(offset,0,length,1) = (param->block(offset,0,length,1)).array() -
                lr_t*scale *NodeFlow::ADAM_M_MAP[asyn_index][offset].array()/((NodeFlow::ADAM_V_MAP[asyn_index][offset]).array().sqrt()+epsilon.array());
    }
    void set_option(double lr=0.001, double beta_1=0.9, double beta_2=0.999)
    {
        lr_=lr;beta_1_=beta_1;beta_2_=beta_2;
    }
};
class GradientDescent:public optimizer
{
private:
    double lr_;
public:
    GradientDescent(double lr=0.001):lr_(lr){}
     /*
     *delta = lr*g
     *w -= delta
     */

    void update(Eigen::MatrixXd* param, Eigen::MatrixXd* gradient,int offset,int length,double scale=1.0,int asyn_index=0)
    {
        param->block(offset,0,length,1) = (param->block(offset,0,length,1)) - lr_*scale*(gradient->block(offset,0,length,1));
    }
    void set_option(double lr=0.01, double prop=0.9, double epsilon=1e-10)
    {
        lr_=lr;
    }
};
class MomentumOptimizer:public optimizer
{
private:
    double lr_;
    double momentum_;
    bool NAG_;
public:
    MomentumOptimizer(double lr=0.001, double momentum=0.9, bool NAG=false):lr_(lr),momentum_(momentum),NAG_(NAG){}
     /*
     *delta = momentum*delta+lr*g
     *w -= delta
     */

    void update(Eigen::MatrixXd* param, Eigen::MatrixXd* gradient,int offset,int length,double scale=1.0,int asyn_index=0)
    {
        if (NodeFlow::MOMENTUM_MAP[asyn_index].find(offset)==NodeFlow::MOMENTUM_MAP[asyn_index].end())
        {
            NodeFlow::MOMENTUM_MAP[asyn_index][offset] = lr_*(gradient->block(offset,0,length,1));
        }else
        {
            NodeFlow::MOMENTUM_MAP[asyn_index][offset]  = momentum_*NodeFlow::MOMENTUM_MAP[asyn_index][offset] + lr_*(gradient->block(offset,0,length,1));
        }
        param->block(offset,0,length,1) = (param->block(offset,0,length,1)).array() -
                scale*NodeFlow::MOMENTUM_MAP[asyn_index][offset].array();
    }
    void set_option(double lr=0.001, double momentum=0.9, double NAG=0)
    {
        lr_=lr;momentum_=momentum;
        if (NAG==0)
            NAG_=false;
        else
            NAG_=true;
    }

};
}
