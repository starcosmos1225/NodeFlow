#pragma once
#include<bits/stdc++.h>
#include "Node.h"
#include "activation_function.h"
#include "inner_function.h"
#include "optimizer.h"
#include "constant.h"
#include "util.h"
#include "omp.h"
//#include "type.h"

namespace NodeFlow
{

//extern std::shared_ptr<NodeFlow::Master> global_master;
//Layer is a base class, it could be denseLayer,CnnLayer or PoolLayer and so on.
//In version 1.0 we only create a denseLayer for full connection
class Layer:public std::enable_shared_from_this<NodeFlow::Layer>
{
private:
    int node_number_;
    std::string name_;
    std::vector<int > shape_;
    double scale_;
    int asyn_index_;
    double dropout_prop_;
public:
    std::shared_ptr<NodeFlow::optimizer> opt_;
    std::shared_ptr<NodeFlow::inner_function> inner_function_;
    std::shared_ptr<NodeFlow::initializer> init_;
    int inner_function_type;
    bool compute_gradient_;
    std::vector<std::shared_ptr<NodeFlow::Node> > node_list;
    Layer(){}
    Layer(int node_number,
          std::shared_ptr<NodeFlow::inner_function> f=nullptr,
          std::shared_ptr<NodeFlow::activation::activation_function> a=nullptr,
          std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
          std::string name = NodeFlow::NONENAME)
    {
        initialze(node_number,f,a,opt,name);
    }
    void initialze(int node_number,
          std::shared_ptr<NodeFlow::inner_function> f=nullptr,
          std::shared_ptr<NodeFlow::activation::activation_function> a=nullptr,
          std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
          std::string name = NodeFlow::NONENAME)
    {
        node_number_ = node_number;
        opt_ = opt;
        name_ = name;
        scale_ = 1.0;
        asyn_index_ = 0;
        inner_function_ = f;
        dropout_prop_ = 0.0;
        if (f&&a)
        {
            f->set_activation_function(a);
        }
    }
    //fp will compute the value and the gradient
    virtual void compute()
    {
        for (auto node:node_list)
        {
            node->compute(compute_gradient_);
        }
    }

    //bp compute the gradient,since gradient may need accumulate for batches, compute_gradient()
    //must separated with update.
    virtual void compute_gradient()
    {
        for (auto node:node_list)
        {
            node->compute_gradient(compute_gradient_);
        }
    }
    //bp will update the trainable param;
    virtual void update()
    {
        for (auto node:node_list)
        {
            node->update(get_scale());
        }
    }
    //get each nodes' value,it can be called to get the results
    std::vector<double> get_value()
    {
        std::vector<double> v_list(node_list.size(),0);
        for (int i=0;i<node_list.size();++i)
        {
            v_list[i] = node_list[i]->get_value();
        }
        return v_list;
    }
    //set each nodes' value ,it can be called to set the input nodes' value
    bool set_value(std::vector<double> &v)
    {
        //clock_t start=clock();
        if (v.size()!=node_list.size())
        {
            std::cout<<"error"<<std::endl;
            return false;
        }
        for (int i=0;i<node_list.size();++i)
        {
            node_list[i]->set_value(v[i]);
        }
        //std::cout<<"inner set data time:"<<(double)(clock()-start)/CLOCKS_PER_SEC<<"s"<<std::endl;
    }
    std::vector<int>& get_shape()
    {
        return shape_;
    }
    void set_shape(std::vector<int> shape)
    {
        shape_ = shape;
    }
    std::vector<std::shared_ptr<Node> >& get_node_list()
    {
        return node_list;
    }
    bool get_activate()
    {
        return false;
    }
    //version 3.0 set the delta to 1.0 and make the layer to the last layer foreward propogation.
    void set_final_delta()
    {
        for (int i=0;i<node_list.size();++i)
        {
            node_list[i]->set_delta(1.0);
            //node_list[i]->set_constant_delta(true);
        }
    }
    int get_node_length()
    {
        return node_number_;
    }
    void set_node_length(int length)
    {
        node_number_=length;
    }
    void stop_gradient()
    {
        compute_gradient_ = false;
    }
    void start_gradient()
    {
        compute_gradient_ = true;
    }
    std::string get_name()
    {
        return name_;
    }
    int get_inner_function()
    {
        return inner_function_type;
    }
    void set_inner_function(int inner_function_type_)
    {
        inner_function_type = inner_function_type_;
    }
    std::shared_ptr<optimizer> get_optimizer()
    {
        return opt_;
    }
    void add_activation_function(std::shared_ptr<NodeFlow::activation::activation_function> a)
    {
        if (inner_function_type!=NodeFlow::MAX_FUNCTION)
        {
            inner_function_->set_activation_function(a);
        }else
        {//MAX_FUNCTION has its own inner function for each node
            for (auto node:node_list)
            {
                node->set_activation_function(a);
            }
        }
        /**/
    }
    //version 3.0 layer scale
    double get_scale()
    {
        return scale_;
    }
    void set_scale(double scale)
    {
        scale_=scale;
    }
    //version 3.0 get param number
    virtual int get_param_number()
    {
        int ans = 0;
        for (auto node:node_list)
        {
            ans += node->get_param_number();
        }
        return ans;
        //realize in the children class
    }
    virtual void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset)
    {
        for (auto node:node_list)
        {
            node->set_param_gradient(param,gradient,offset);
        }
        //realize in the children class
    }
    void initialize()
    {
        for (auto node:node_list)
        {
            node->initialize_param(init_);
        }
        return;
    }
    virtual int get_asyn_index()
    {
        return asyn_index_;
    }
    virtual void set_asyn_index(int asyn_index)
    {
        asyn_index_ = asyn_index;
    }
    //set dropout prop
    void set_dropout(double dropout_prop)
    {
        dropout_prop_ = dropout_prop;
    }
    double get_dropout()
    {
        return dropout_prop_;
    }
    //save and load
    void save(std::ofstream& savefile)
    {
        return;
    }
    void load(std::ifstream& loadfile,int n)
    {
        return;
    }
    virtual void display()
    {
        for (auto node:node_list)
        {
            node->display(1);
        }
        return;
    }
};

class DenseLayer:public Layer
{
private:
    int param_number_;
    int input_length_;
    int offset_;
    Eigen::MatrixXd* dense_param;
    Eigen::MatrixXd* dense_gradient;
    std::vector<int> dropout_list;
public:
    DenseLayer(std::shared_ptr<NodeFlow::Layer> input,
               int node_number,
               NodeFlow::initializer &init = nf_constant,
               std::shared_ptr<NodeFlow::inner_function> f=nullptr,
               std::shared_ptr<NodeFlow::activation::activation_function> a=nullptr,
               std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
               std::string name = NodeFlow::NONENAME)
    {
        init_ =init.make_ptr();
        //std::cout<<"init:"<<init_->init()<<std::endl;
        initialze(node_number,f,a,opt,name);
        //NodeFlow::global_master->build_link(input_name,get_name());
        compute_gradient_ = true;
        std::vector<std::shared_ptr<Node> > input_nodes = input->get_node_list();
        input_length_ = input_nodes.size();
        param_number_ = node_number;//bias number
        for (int i=0;i<node_number;++i)
        {
            std::shared_ptr<NodeFlow::Node> new_node =
                    std::make_shared<NodeFlow::Node>(f,opt);
            node_list.push_back(new_node);
            for (auto input_node:input_nodes)
            {
                param_number_ ++;
                input_node->connect(node_list[i],NodeFlow::OUTNODE);
                node_list[i]->connect(input_node,NodeFlow::INNODE);
            }
        }
        //std::cout<<"dense param:"<<param_number_<<std::endl;
        set_shape(std::vector<int>{node_number,1});
    }
    //This constructor only used when optimizer_function is PRelu
    DenseLayer(std::shared_ptr<NodeFlow::Layer> input,
               int node_number,
               NodeFlow::initializer init,
               std::shared_ptr<NodeFlow::inner_function> f,
               int activation_function_name,
               std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
                std::string name = NodeFlow::NONENAME)
    {
        if (activation_function_name==NodeFlow::PRELU)
        {
            new (this)DenseLayer(input,node_number,init,f,nullptr,opt,name);
            for (auto node:node_list)
            {
                node->set_activation_function(std::make_shared<NodeFlow::activation::PRelu>());
            }
        }else
        {
            //raise error
            throw "the constructor only used for PRelu but the activation_function is not PRelu";
        }

    }


    int get_param_number()
    {
        int ans = 0;
        for (auto node:node_list)
        {
            ans+=node->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset)
    {
        dense_param = param;
        dense_gradient = gradient;
        offset_ = offset;
        for (auto node:node_list)
        {
            node->set_param_gradient(param,gradient,offset);
        }
        //realize in the children class
    }
    /*void initialize()
    {
        for (int i=offset_;i<offset_+param_number_;++i)
        {
            if ((i-offset_) % input_length_!=0)
                (*dense_param)(i,0) = init_->init();
            else
            {
                (*dense_param)(i,0) = 0.0;
            }
            (*dense_gradient)(i,0) = 0.0;
        }
    }*/
    void update()
    {
        if (opt_)
        {
            //std::cout<<"begin update:"<<get_asyn_index()<<std::endl;
            opt_->update(dense_param,dense_gradient,offset_,param_number_,get_scale(),get_asyn_index());
             //std::cout<<"end update"<<std::endl;
        }
        //std::cout<<"begin node update:"<<std::endl;
         for (auto node:node_list)
        {
            node->update(1.0,false);
        }
        //std::cout<<"end node update:"<<std::endl;
    }
    //we only set dense layer to dropout
    void compute()
    {
        double p = rand()*1.0/INT_MAX;
        double prop = get_dropout();
        dropout_list.clear();
        for (int i=0;i<node_list.size();++i)
        {
            if (p>= prop)
                node_list[i]->compute(compute_gradient_);
            else
            {
                node_list[i]->set_value(0.0);
                dropout_list.push_back(i);
            }
        }
    }
    void compute_gradient()
    {
        int index=0;
        for (int i=0;i<node_list.size();++i)
        {
            if (index<dropout_list.size()&&i==dropout_list[index])
            {
                index++;
            }else
                node_list[i]->compute_gradient(compute_gradient_);
        }
    }
    void save(std::ofstream &savefile)
    {
        int n = get_param_number();
        savefile<<get_name()<<" "<<n<<std::endl;
        for (int i=offset_;i<offset_+n;++i)
        {
            savefile<<(*dense_param)(i,0)<<" ";
        }
        savefile<<std::endl;
        return;
    }
    void load(std::ifstream &loadfile,int n)
    {
        for (int i=offset_;i<offset_+n;++i)
        {
            loadfile>>(*dense_param)(i,0);
        }
        return;
    }
    void display()
    {
        for (auto node:node_list)
        {
           node->display(0);
        }
    }
};
class SparseLayer:public Layer
{
private:
    int param_number_;
    int input_length_;
    int offset_;
    Eigen::MatrixXd* sparse_param;
    Eigen::MatrixXd* sparse_gradient;
    std::vector<int> dropout_list;
public:
    SparseLayer(std::shared_ptr<NodeFlow::Layer> input,
                int size,
                int step,
                NodeFlow::initializer &init = nf_constant,
                std::shared_ptr<NodeFlow::inner_function> f=nullptr,
                std::shared_ptr<NodeFlow::activation::activation_function> a=nullptr,
                std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
                std::string name = NodeFlow::NONENAME)
    {
        init_ =init.make_ptr();

        compute_gradient_ = true;
        std::vector<std::shared_ptr<Node> > input_nodes = input->get_node_list();
        int node_number = 0;
        for (int i=0;i<input_nodes.size()-size + 1;i+=step)
        {
            node_number ++;
        }
        std::cout<<"node_number:"<<node_number<<std::endl;
        initialze(node_number,f,a,opt,name);
        input_length_ = input_nodes.size();
        param_number_ = node_number;//bias number
        for (int i=0;i<node_number;++i)
        {
            std::shared_ptr<NodeFlow::Node> new_node =
                    std::make_shared<NodeFlow::Node>(f,opt);
            node_list.push_back(new_node);
            for (int j=i;j<i+size;j++)
            {
                param_number_++;
                input_nodes[j]->connect(node_list[i],NodeFlow::OUTNODE);
                node_list[i]->connect(input_nodes[j],NodeFlow::INNODE);
            }
        }
        set_shape(std::vector<int>{node_number,1});
    }
    //This constructor only used when optimizer_function is PRelu
    SparseLayer(std::shared_ptr<NodeFlow::Layer> input,
               int size,
               int step,
               NodeFlow::initializer init,
               std::shared_ptr<NodeFlow::inner_function> f,
               int activation_function_name,
               std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
                std::string name = NodeFlow::NONENAME)
    {
        if (activation_function_name==NodeFlow::PRELU)
        {
            new (this)SparseLayer(input,size,step,init,f,nullptr,opt,name);
            for (auto node:node_list)
            {
                node->set_activation_function(std::make_shared<NodeFlow::activation::PRelu>());
            }
        }else
        {
            //raise error
            throw "the constructor only used for PRelu but the activation_function is not PRelu";
        }

    }


    int get_param_number()
    {
        int ans = 0;
        for (auto node:node_list)
        {
            ans+=node->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset)
    {
        sparse_param = param;
        sparse_gradient = gradient;
        offset_ = offset;
        for (auto node:node_list)
        {
            node->set_param_gradient(param,gradient,offset);
        }
        //realize in the children class
    }
    void update()
    {
        if (opt_)
        {
            opt_->update(sparse_param,sparse_gradient,offset_,param_number_,get_scale(),get_asyn_index());
        }
         for (auto node:node_list)
        {
            node->update(1.0,false);
        }
    }
    void compute()
    {
        double p = rand()*1.0/INT_MAX;
        double prop = get_dropout();
        dropout_list.clear();
        for (int i=0;i<node_list.size();++i)
        {
            if (p>= prop)
                node_list[i]->compute(compute_gradient_);
            else
            {
                node_list[i]->set_value(0.0);
                dropout_list.push_back(i);
            }
        }
    }
    void compute_gradient()
    {
        int index=0;
        for (int i=0;i<node_list.size();++i)
        {
            if (index<dropout_list.size()&&i==dropout_list[index])
            {
                index++;
            }else
                node_list[i]->compute_gradient(compute_gradient_);
        }
    }
    void save(std::ofstream &savefile)
    {
        int n = get_param_number();
        savefile<<get_name()<<" "<<n<<std::endl;
        for (int i=offset_;i<offset_+n;++i)
        {
            savefile<<(*sparse_param)(i,0)<<" ";
        }
        savefile<<std::endl;
        return;
    }
    void load(std::ifstream &loadfile,int n)
    {
        for (int i=offset_;i<offset_+n;++i)
        {
            loadfile>>(*sparse_param)(i,0);
        }
        return;
    }
    void display()
    {
        for (auto node:node_list)
        {
           node->display(0);
        }
    }
};
//since we use a conception of node ,so there is no padding option in the CNNLayer, because
//we can not explain the padding operation's significance in physic world.
//CNN is special for the param. param in node is shared by each other.
//version2.0 for cnn

class CNNLayer:public Layer
{
private:
    int param_number_;
    int depth_;
    int offset_;
    Eigen::MatrixXd* cnn_param;
    Eigen::MatrixXd* cnn_gradient;
public:
    CNNLayer(std::shared_ptr<NodeFlow::Layer> input,
             std::vector<int> CNN_shape,//[x,y,z]:x,y is the feature map's shape ,and z is the number of feature map
             NodeFlow::initializer &init,
             std::shared_ptr<NodeFlow::inner_function> f=nullptr,
             std::shared_ptr<NodeFlow::activation::activation_function> a=nullptr,
             std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
             std::string name = NodeFlow::NONENAME)

    {
        init_ =init.make_ptr();
        compute_gradient_ = true;
        std::vector<int> input_shape = input->get_shape();

        int row,col,depth,input_depth,rc;
        row = input_shape[0]-CNN_shape[0]+1;
        col = input_shape[1]-CNN_shape[1]+1;
        rc = input_shape[0]*input_shape[1];
        depth = CNN_shape[2];
        int node_number = row*col*depth;
        initialze(node_number,f,a,opt,name);
        set_shape(std::vector<int>{row,col,depth});
        if (input_shape.size()<=2)
        {
            input_depth = 1;
        }else
        {
            input_depth = input_shape[2];
        }
        param_number_ = (CNN_shape[0]*CNN_shape[1]*input_depth+1)*depth;
        depth_ = depth;
        //the convolution param is cnn_row*cnn_col*input_shape[2]*CNN_shape[2]
        //each row*col*input_shape[2] param dot the part of input and construct one outputnode,
        //the node_number = (input[0]-cnn[0]+1)*(input[1]-cnn[1]+1)*cnn[2]
        //opt only need save in the cnn layer
        //param is row_wise
        std::vector<std::shared_ptr<Node> > input_nodes = input->get_node_list();
        int rc_= row*col;
        for (int i=0;i<node_number;++i)
        {
            //for each node i, the location is z=i/(row*col),x = i / z / col y=i / z % col
            int z = i / rc_;
            int x = (i % rc_) / col;
            int y = (i % rc_) % col;
            //z is the index of param[]->param[z]
            //x,y is the begin of the input node. that we connect from [x,y,0] to
            //[x+cnn_x-1,y+cnn_y-1,input_depth-1]
            std::shared_ptr<NodeFlow::Node> new_node = std::make_shared<NodeFlow::Node>
                    (f,
                     opt);
            node_list.push_back(new_node);
            //for (int xx = x;xx<x+CNN_shape[0];xx++)
            for (int zz = 0;zz<input_depth;zz++)
            {
                //for (int yy = y;yy<y+CNN_shape[1];yy++)
                for (int xx = x;xx<x+CNN_shape[0];xx++)
                {
                    for (int yy = y;yy<y+CNN_shape[1];yy++)
                    {
                        //for input_nodes:the [xx,yy,zz] map->[zz*(row*col)+xx*(col)+yy]
                        int index = zz*rc+xx*input_shape[1]+yy;
                        if (index<input_nodes.size())
                        {
                            input_nodes[index]->connect(node_list[i],NodeFlow::OUTNODE);
                            node_list[i]->connect(input_nodes[index],NodeFlow::INNODE);
                        }else
                        {
                            throw "index out of input shape!";
                        }
                    }
                }
            }
        }
    }
    //This constructor only used when optimizer_function is PRelu
    CNNLayer(std::shared_ptr<NodeFlow::Layer> input,
             std::vector<int> CNN_shape,//[x,y,z]:x,y is the feature map's shape ,and z is the number of feature map
             NodeFlow::initializer &init,
             std::shared_ptr<NodeFlow::inner_function> f,
             int activation_function_name,
             std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
             std::string name = NodeFlow::NONENAME)
    {
        if (activation_function_name==NodeFlow::PRELU)
        {
            new (this)CNNLayer(input,CNN_shape,init,f,nullptr,opt,name);
            for (auto node:node_list)
            {
                node->set_activation_function(std::make_shared<NodeFlow::activation::PRelu>());
            }
        }else
        {
            //raise error
            throw "the constructor only used for PRelu but the activation_function is not PRelu";
        }

    }
    void compute()
    {
        for (auto node:node_list)
        {
            node->compute(compute_gradient_);
        }
    }
    void compute_gradient()
    {
        for (auto node:node_list)
        {
            node->compute_gradient(compute_gradient_);
        }
    }
    int get_param_number()
    {
        int ans = param_number_;
        for (auto node:node_list)
        {
            ans+=node->get_param_number(true);
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient,int &offset)
    {
        offset_ = offset;
        int param_number = param_number_/depth_;
        int each_layer_number = node_list.size()/depth_;
        cnn_param = param;
        cnn_gradient = gradient;
        for (int i=0;i<node_list.size();++i)
        {
            //there are param_number_'s node param and some activation param
            //the node param need split with depth,each depth has param_number_/depth param
            //so the k=i/depth need to offset the k*(i/depth)
            //int depth_ = i/each_layer_number;
            if (i!=0 && i% each_layer_number==0)
            {
                offset += param_number;
            }
            node_list[i]->set_param_gradient(param,gradient,offset,param_number,true,false);//true means cnn
        }
        offset += param_number;
        for (int i=0;i<node_list.size();++i)
        {
            //set the inner function 's param(only prelu need this)
            node_list[i]->set_param_gradient(param,gradient,offset,each_layer_number,true,true);
        }
    }
    //the update must in the cnn layer
    void update()
    {
        if (opt_)
        {
            opt_->update(cnn_param,cnn_gradient,offset_,param_number_,get_scale(),get_asyn_index());
        }
         for (auto node:node_list)
        {
            node->update(1.0,false);
        }
    }
    /*void initialize()
    {
        int param_number = param_number_/depth_;
        for (int i=offset_;i<offset_+param_number_;++i)
        {
            if ((i-offset_)%param_number!=0)
            {
                (*cnn_param)(i,0) = init_->init();
            }else
                (*cnn_param)(i,0) = 0.0;

            (*cnn_gradient)(i,0) = 0.0;
        }
    }*/
    //save and load
    void save(std::ofstream& savefile)
    {
        int n = get_param_number();
        savefile<<get_name()<<" "<<n<<std::endl;
        for (int i=offset_;i<offset_+n;++i)
        {
            savefile<<(*cnn_param)(i,0)<<" ";
        }
        savefile<<std::endl;
        return;
    }
    void load(std::ifstream& loadfile,int n)
    {
        for (int i=offset_;i<offset_+n;++i)
        {
            loadfile>>(*cnn_param)(i,0);
        }
        return;
    }

};
//version2.0 for maxpooling,pooling has no param to train, so it is similar as MathLayer
class MaxPoolingLayer:public Layer
{
public:
    MaxPoolingLayer(std::shared_ptr<NodeFlow::Layer> input,
                    std::vector<int> pooling_shape,//[x,y]:the pooling size
                    int step_size,//the skip size
                    //the inner_function is different with Normal_function
                    //not like:v = w*x+b
                    //it like:v = max(x),the function call max_function
                    //bp is max(1)
                    std::string name = NodeFlow::NONENAME)

    {
        initialze(0, nullptr, nullptr, nullptr, name);
        std::vector<int> input_shape = input->get_shape();
        int depth = input_shape.size()<=2?1:input_shape[2];
        std::vector<std::shared_ptr<Node> > input_nodes = input->get_node_list();
        int row = input_shape[0];
        int col = input_shape[1];
        int rc= row*col;
        int node_number = 0;
        set_shape(std::vector<int>{row/step_size,col/step_size,depth});
        for (int k=0;k<depth;k++)
        {
            for (int i=0;i<input_shape[0];i+=step_size)
            {
                for (int j=0;j<input_shape[1];j+=step_size)
                {

                    //create a new node and push into the node_list
                    std::shared_ptr<NodeFlow::Node> new_node = std::make_shared<NodeFlow::Node>
                    (std::make_shared<NodeFlow::Max_function>(),
                     nullptr);
                     node_list.push_back(new_node);

                    //connect the node with[i,j,k]->[i+pooling_shape[0],j+pooling_shape[0],k]
                    for (int xx = i;xx<i+pooling_shape[0];xx++)
                    {
                        for (int yy=j;yy<j+pooling_shape[1];yy++)
                        {
                            //the [i,j,k]->[k*row*col+i*col+j]
                            int index = k*rc+xx*col+yy;
                            if (index<input_nodes.size())
                            {
                                input_nodes[index]->connect(node_list[node_number],NodeFlow::OUTNODE);
                                node_list[node_number]->connect(input_nodes[index],NodeFlow::INNODE);
                            }
                        }
                    }
                    node_number++;
                }
            }
        }
        set_node_length(node_number);
        compute_gradient_ = false;
    }
    int get_param_number()
    {
        return 0;
    }
    void update()
    {
        return;
    }
};
//version2.0 for meanpooling
class MeanPoolingLayer:public Layer
{
public:
    MeanPoolingLayer(std::shared_ptr<NodeFlow::Layer> input,
                    std::vector<int> pooling_shape,//[x,y]:the pooling size
                    int step_size,//the skip size
                    std::shared_ptr<NodeFlow::inner_function> f,//the inner_function is different with Normal_function
                    //not like:v = w*x+b
                    //it like:v = max(x),the function call max_function
                    //bp is max(1)
                    std::string name = NodeFlow::NONENAME)

    {
        initialze(0, nullptr, nullptr, nullptr, name);
        std::vector<int> input_shape = input->get_shape();
        //node_number = input_shape/pooling_shape:input_shape[0]/pooling_shape[0],input_shape[1]/pooling_shape[1]
        int depth = input_shape.size()<=2?1:input_shape[2];
        std::vector<std::shared_ptr<Node> > input_nodes = input->get_node_list();
        int row = input_shape[0];
        int col = input_shape[1];
        int rc= row*col;
        int node_number = 0;
        set_shape(std::vector<int>{row/step_size,col/step_size,depth});
        for (int k=0;k<depth;k++)
        {
            for (int i=0;i<input_shape[0];i+=step_size)
            {
                for (int j=0;j<input_shape[1];j+=step_size)
                {

                    //create a new node and push into the node_list
                    std::shared_ptr<NodeFlow::Node> new_node = std::make_shared<NodeFlow::Node>
                    (f,
                     nullptr);
                     node_list.push_back(new_node);

                    //connect the node with[i,j,k]->[i+pooling_shape[0],j+pooling_shape[0],k]
                    for (int xx = i;xx<i+pooling_shape[0];xx++)
                    {
                        for (int yy=j;yy<j+pooling_shape[1];yy++)
                        {
                            //the [i,j,k]->[k*row*col+i*col+j]
                            int index = k*rc+xx*col+yy;
                            if (index<input_nodes.size())
                            {
                                input_nodes[index]->connect(node_list[node_number],NodeFlow::OUTNODE);
                                node_list[node_number]->connect(input_nodes[index],NodeFlow::INNODE);
                            }
                        }
                    }
                    node_number++;
                }
            }
        }
        set_node_length(node_number);
        compute_gradient_ = false;
    }
    void update()
    {
        return;
    }
};

class MathLayer:public Layer
{
public:
    MathLayer(int node_number,
              std::shared_ptr<NodeFlow::inner_function> f=nullptr,
              std::shared_ptr<NodeFlow::activation::activation_function> a=nullptr,
              std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
              std::string name = NodeFlow::NONENAME):
               Layer(node_number, f, a, opt, name)
    {
        for (int i=0;i<node_number;++i)
        {
            std::shared_ptr<Node> new_node =
                    std::make_shared<NodeFlow::Node>(f, opt);
            node_list.push_back(new_node);
        }
        compute_gradient_ = false;
    }
    void update()
    {
        return;
    }
};

class InputLayer:public Layer
{
public:
    InputLayer(int node_number,std::vector<int> &shape,std::string name):
        Layer(node_number,nullptr,nullptr,nullptr,name)
    {
        if (shape.size()>0)
            set_shape(shape);
        else
        {
            std::vector<int> tmp_shape({node_number});
            set_shape(tmp_shape);
        }

        for (int i=0;i<node_number;++i)
        {
            node_list.push_back(std::make_shared<NodeFlow::Node>());
        }
    }
    //fp and bp are no need for InputLayer
    void compute(){return;}
    void compute_gradient(){return;}
    void update(){return;}
    int get_param_number()
    {
        return 0;
    }
    void set_param_gradient(Eigen::MatrixXd *param,Eigen::MatrixXd *gradient,int &offset)
    {

        return;
    }
};
}
