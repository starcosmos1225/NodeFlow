#pragma once
#include <bits/stdc++.h>

namespace NodeFlow
{
    double uniform_random()
    {
        return rand()*1.0/INT_MAX;
    }
    double guass_random(double m=0,double delta=1.0)
    {
        double u1 = NodeFlow::uniform_random();
        double u2 = NodeFlow::uniform_random();
        double z = sqrt(-2*log(u1))*sin(2*M_PI*u2);
        return z*delta+m;
    }
    double glorot_uniform(double fan_in,double fan_out)
    {
        double limit = sqrt(6.0/(fan_in+fan_out));
        double v = -limit+NodeFlow::uniform_random()*2*limit;
        return v;
    }
    double constant(double m=0,double delta=1.0)
    {
        return m;
    }
    std::vector<int> create_random_int(int number,int range)
    {
        std::vector<int> n_list;
        for (int i=0;i<range;i++)
        {
            n_list.push_back(i);
        }
        std::vector<int> ans;
        int l = range;
        for (int i=0;i<number;++i)
        {
            int index = rand() % l;
            ans.push_back(n_list[index]);
            n_list[index] = n_list[l-1];
            l--;
        }
        return ans;
    }
    std::vector<double> onehot(unsigned int n,int range)
    {
        std::vector<double> ans(range,0);
        ans[n] = 1.0;
        return ans;
    }
    int argmax(std::vector<double> n)
    {
        int ans;
        double max_=DBL_MIN;
        for (int i=0;i<n.size();i++)
        {
            if (max_<n[i])
            {
                ans=i;
                max_=n[i];
            }
        }
        return ans;
    }
class initializer
{
public:
    double a_;
    double b_;
    initializer(double a=0,double b=1.0):a_(a),b_(b){}
    virtual double init(){}
    virtual std::shared_ptr<initializer> make_ptr(){return nullptr;}
};
class Constant:public initializer
{
public:
    Constant(double a=0,double b=1.0):initializer(a,b){}
    double init(){
        return a_;
        }
    std::shared_ptr<initializer> make_ptr()
    {
        return std::shared_ptr<initializer>((initializer* )new Constant(a_,b_));
    }
};
class Guass:public initializer
{
public:
    Guass(double a=0,double b=1.0):initializer(a,b){}
    double init(){
        return NodeFlow::guass_random(a_,b_);}
    std::shared_ptr<initializer> make_ptr()
    {
        return std::shared_ptr<initializer>((initializer* )new Guass(a_,b_));
    }
};
class Glorot:public initializer
{
public:
    Glorot(double a=0,double b=1.0):initializer(a,b){}
    double init(){
        return NodeFlow::glorot_uniform(a_,b_);}
    std::shared_ptr<initializer> make_ptr()
    {
        return std::shared_ptr<initializer>((initializer* )new Glorot(a_,b_));
    }
};
}
