#ifndef MASTER_H_INCLUDED
#define MASTER_H_INCLUDED
#include "Layer.h"
#include "activation_function.h"
#include "optimizer.h"
#include "Graph.h"
#include "util.h"
#include<bits/stdc++.h>
#include "omp.h"
namespace NodeFlow
{
//Master is the main controller of the NodeFlow, it control the Layer's connect and regist the layer with its name
//extern int NodeFlow::Adam_iteration;
class Master
{
private:
    //std::unordered_map<int, std::shared_ptr<NodeFlow::inner_function> > Inner_function_list;
    std::unordered_map<int, std::shared_ptr<NodeFlow::activation::activation_function> > Activation_function_list;
    std::unordered_map<std::string, std::shared_ptr<NodeFlow::Layer> > layer_map_;
    Eigen::MatrixXd param_;
    Eigen::MatrixXd gradient_;
public:
    Master()
    {
        //Inner_function_list[NORMAL_FUNCTION] = std::make_shared<inner_function>();
        //Inner_function_list[MEAN_FUNCTION] = std::make_shared<Mean_function>();
        //Inner_function_list[ADD_FUNCTION] = std::make_shared<Add_function>();
        //Inner_function_list[MINUS_FUNCTION] = std::make_shared<Minus_function>();

        Activation_function_list[NodeFlow::NONE] = nullptr;
        Activation_function_list[NodeFlow::SIGMOID] = std::make_shared<NodeFlow::activation::sigmoid>();
        Activation_function_list[NodeFlow::SQUARE] = std::make_shared<NodeFlow::activation::square>();
        Activation_function_list[NodeFlow::TANH] = std::make_shared<NodeFlow::activation::tanh>();
        Activation_function_list[NodeFlow::RELU] = std::make_shared<NodeFlow::activation::Relu>();
        Activation_function_list[NodeFlow::LEAKYRELU] = std::make_shared<NodeFlow::activation::LeakyRelu>();
        //PRelu could not create in here. It must create in layer.
        Activation_function_list[NodeFlow::SQUARE] = std::make_shared<NodeFlow::activation::square>();
    }
    void regist_layer(std::shared_ptr<NodeFlow::Layer> layer,std::string name)
    {
        if (layer_map_.find(name)==layer_map_.end())
        {
            layer_map_[name] = layer;
        }else
        {
            //TODO raise a error
            throw "The layer is already in the Master";
        }
    }
    std::shared_ptr<NodeFlow::Layer> get_layer(std::string name)
    {
        return layer_map_[name];
    }
    //set the layer not to compute the gradient
    void stop_gradient(std::string layer)
    {
        layer_map_[layer]->stop_gradient();
    }
    //set the layer compute the gradient
    void start_gradient(std::string layer)
    {
        layer_map_[layer]->start_gradient();
    }
    void stop_all_gradient()
    {
        for (auto iter=layer_map_.begin();iter!=layer_map_.end();++iter)
        {
            iter->second->stop_gradient();
        }
    }
    void start_all_gradient()
    {
        for (auto iter=layer_map_.begin();iter!=layer_map_.end();++iter)
        {
            iter->second->start_gradient();
        }
    }
    //get the layer's value
    std::vector<double> get_value(std::string layer)
    {
        if (layer_map_.find(layer)==layer_map_.end())
        {
            throw "The layer name not found!";
        }
        return layer_map_[layer]->get_value();
    }
    //set the layer's value,and it will activate the layer
    void set_value(std::string layer, std::vector<double> &v)
    {
        layer_map_[layer]->set_value(v);
    }
    //compute the layer's value and the gradient value
    std::vector<std::vector<double> > compute(std::vector<std::string> &layer_list,
                                              std::string layer,
                                              std::unordered_map<std::string,std::vector< std::vector<double> > > &dict,
                                              bool evaluate=false)
    {
        int datalength = dict.begin()->second.size();
        std::vector<std::vector<double> > result;
        for (int i=0;i<datalength;++i)
        {

            for (auto iter = dict.begin();iter!=dict.end();iter++)
            {
                std::string layer_name = iter->first;
                set_value(layer_name,(iter->second)[i]);
            }
            for (std::string layer_name:layer_list)
            {
                layer_map_[layer_name]->compute();

            }
            std::vector<double> loss = get_value(layer);
            result.push_back(loss);
            //if (evaluate)
                //continue;

            layer_map_[layer]->set_final_delta();
            for (int j=layer_list.size()-1;j>=0;--j)
            {
                layer_map_[layer_list[j]]->compute_gradient();
            }

        }
        return result;
    }
    void minimize(std::string layer,std::vector<std::string> &layer_list)
    {
        //std::cout<<"begin minimize"<<std::endl;
        for (int i=layer_list.size()-1;i>=0;--i)
        {
            //std::cout<<"layer"<<layer_list[i]<<std::endl;
            get_layer(layer_list[i])->update();
        }
        //std::cout<<"end minimize"<<std::endl;
        gradient_.setZero();
    }
    //get the layer's name through the layer's point
    std::string get_layer_name(std::shared_ptr<Layer> layer)
    {
        //std::shared_ptr<Layer> layer_ptr = layer.get_layer_ptr();
        for (auto iter = layer_map_.begin();iter!=layer_map_.end();++iter)
        {
            if (iter->second == layer)
                return iter->first;
        }
        //TODO raise error:not found a layer name.
        // Maybe it is not a error.It could be a warning
        return NodeFlow::NONENAME;
    }
    /*create Layers*/
    std::shared_ptr<DenseLayer> create_DenseLayer(std::string input ,
               int node_number,
               NodeFlow::initializer &init=NodeFlow::nf_constant,
               int inner_function_name=NodeFlow::NORMAL_FUNCTION,//inner_function
               int activation_function_name=NodeFlow::NONE,//nullptr
               std::shared_ptr<optimizer> opt=nullptr,
               std::string name = NodeFlow::NONENAME)
    {
        auto input_ = get_layer(input);
        if (inner_function_name==NodeFlow::NORMAL_FUNCTION)
        {
            std::string name_=name;
            std::shared_ptr<DenseLayer> layer;
            if (activation_function_name!=NodeFlow::PRELU)
            {
                layer = std::make_shared<DenseLayer>(input_,
                                                     node_number,
                                                     init,
                                                     std::make_shared<inner_function>(),
                                                     Activation_function_list[activation_function_name],
                                                     opt,
                                                     name_);
            }else
            {
                layer = std::make_shared<DenseLayer>(input_,
                                                     node_number,
                                                     init,
                                                     std::make_shared<inner_function>(),
                                                     activation_function_name,
                                                     opt,
                                                     name_);
            }
            layer->set_inner_function(inner_function_name);
            regist_layer(layer,name_);
            return layer;
        }else
        {
            //TODO raise error
            throw "The inner function must be normal function!";
        }

    }
    std::shared_ptr<SparseLayer> create_SparseLayer(std::string input ,
                                                    int size,
                                                    int step,
                                                    NodeFlow::initializer &init=NodeFlow::nf_constant,
                                                    int inner_function_name=NodeFlow::NORMAL_FUNCTION,//inner_function
                                                    int activation_function_name=NodeFlow::NONE,//nullptr
                                                    std::shared_ptr<optimizer> opt=nullptr,
                                                    std::string name = NodeFlow::NONENAME)
    {
        auto input_ = get_layer(input);
        if (inner_function_name==NodeFlow::NORMAL_FUNCTION)
        {
            std::string name_=name;
            std::shared_ptr<SparseLayer> layer;
            if (activation_function_name!=NodeFlow::PRELU)
            {
                layer = std::make_shared<SparseLayer>(input_,
                                                      size,
                                                      step,
                                                      init,
                                                      std::make_shared<inner_function>(),
                                                      Activation_function_list[activation_function_name],
                                                      opt,
                                                      name_);
            }else
            {
                layer = std::make_shared<SparseLayer>(input_,
                                                      size,
                                                      step,
                                                      init,
                                                      std::make_shared<inner_function>(),
                                                      activation_function_name,
                                                      opt,
                                                      name_);
            }
            layer->set_inner_function(inner_function_name);
            regist_layer(layer,name_);
            return layer;
        }else
        {
            //TODO raise error
            throw "The inner function must be normal function!";
        }

    }
    void create_CNNLayer(
            std::string input ,
            std::vector<int> cnn_shape,
            NodeFlow::initializer &init=NodeFlow::nf_constant,
            int inner_function_name=NodeFlow::NORMAL_FUNCTION,//inner_function
            int activation_function_name=NodeFlow::NONE,//nullptr
            std::shared_ptr<optimizer> opt=nullptr,
            std::string name = NodeFlow::NONENAME)
    {
        auto input_ = get_layer(input);
        //std::cout<<init.init()<<std::endl;
        //int t;
        //std::cin>>t;
        if (inner_function_name==NodeFlow::NORMAL_FUNCTION)
        {
            std::string name_=name;
            std::shared_ptr<CNNLayer> layer;
            if (activation_function_name!=NodeFlow::PRELU)
            {
                layer = std::make_shared<CNNLayer>(input_,
                                                   cnn_shape,
                                                   init,
                                                   std::make_shared<inner_function>(),
                                                   Activation_function_list[activation_function_name],
                                                   opt,
                                                   name_);
            }else
            {

                layer = std::make_shared<CNNLayer>(input_,
                                                   cnn_shape,
                                                   init,
                                                   std::make_shared<inner_function>(),
                                                   activation_function_name,
                                                   opt,
                                                   name_);
            }
            layer->set_inner_function(inner_function_name);
            regist_layer(layer,name_);
        }else
        {
            //TODO raise error
            throw "The inner function must be normal function!";
        }

    }
    std::shared_ptr<Layer> create_poolingLayer(
            std::string input ,
            std::vector<int> pooling_shape,
            int step_size=2,
            int inner_function_name=NodeFlow::MAX_FUNCTION,//inner_function
            std::string name = NodeFlow::NONENAME)
    {
        auto input_ = get_layer(input);
        if (inner_function_name==NodeFlow::MAX_FUNCTION)
        {
            std::string name_=name;
            std::shared_ptr<MaxPoolingLayer> layer = std::make_shared<MaxPoolingLayer>(input_,pooling_shape,step_size,name_);
            layer->set_inner_function(inner_function_name);
            regist_layer(layer,name_);
            return layer;
        }else if (inner_function_name==NodeFlow::MEAN_FUNCTION)
        {
            std::string name_=name;
            std::shared_ptr<MeanPoolingLayer> layer = std::make_shared<MeanPoolingLayer>(input_,
                                                                                         pooling_shape,
                                                                                         step_size,
                                                                                         std::make_shared<Mean_function>(),
                                                                                         name_);
            layer->set_inner_function(inner_function_name);
            regist_layer(layer,name_);
            return layer;
        }else
        {
            //TODO raise error
            throw "The inner function is not pooling function!";
        }

    }
    void create_InputLayer(int node_number,std::vector<int> &shape,std::string name)
    {
        std::string name_=name;
        std::shared_ptr<InputLayer> layer = std::make_shared<InputLayer>(node_number,shape,name_);
        regist_layer(layer, name_);
    }
    std::shared_ptr<MathLayer> create_MathLayer(int node_number,
                                             int f=NodeFlow::NORMAL_FUNCTION,
                                            int  activation_function_name=NONE,
                                            std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
                                            std::string name = NodeFlow::NONENAME)
    {
        //if (Inner_function_list.find(f)!=Inner_function_list.end())
        //{
            std::shared_ptr<inner_function> f_;
            if (f==NodeFlow::NORMAL_FUNCTION)
            {
                f_=std::make_shared<inner_function>();
            }else if (f==NodeFlow::MAX_FUNCTION)
                f_=std::make_shared<Max_function>();
            else if (f==NodeFlow::ADD_FUNCTION)
                            f_=std::make_shared<Add_function>();
            else if (f==NodeFlow::MINUS_FUNCTION)
                            f_=std::make_shared<Minus_function>();
            else if (f==NodeFlow::MEAN_FUNCTION)
                            f_=std::make_shared<Mean_function>();
            std::string name_=name;
            std::shared_ptr<MathLayer> layer = std::make_shared<MathLayer>(node_number,
                                                                           f_,
                                                                           Activation_function_list[activation_function_name],
                                                                           opt,
                                                                           name_);
            regist_layer(layer, name_);
            return layer;
        //}else
        //{
            //TODO raise error
            //throw "The inner function is not exist!";
        //}

    }


    /*MathLayer add minus square*/
    //A+B->C
    std::shared_ptr<Layer> add(std::string layer_A,std::string layer_B, std::string layer_C)
    {
        auto A = get_layer(layer_A);
        auto B = get_layer(layer_B);
        auto C = get_layer(layer_C);
        if (A->get_node_length()!=B->get_node_length())
        {
            //TODO:raise error
            throw "The operator's two layer 's length are not same! But the operator need same length";
        }else
        {
            int node_number = A->get_node_length();
            //std::shared_ptr<MathLayer> C = create_MathLayer(node_number, A->get_inner_function());
            std::vector<std::shared_ptr<Node> > input_node_A = A->get_node_list();
            std::vector<std::shared_ptr<Node> > input_node_B = B->get_node_list();
            std::vector<std::shared_ptr<Node> > node_list = C->get_node_list();
            for (int i=0;i<node_number;++i)
            {
                input_node_A[i]->connect(node_list[i],NodeFlow::OUTNODE);
                input_node_B[i]->connect(node_list[i],NodeFlow::OUTNODE);
                node_list[i]->connect(input_node_A[i],NodeFlow::INNODE);
                node_list[i]->connect(input_node_B[i],NodeFlow::INNODE);
            }
            return C;
        }
    }
    //A-B->C
    std::shared_ptr<Layer> minus(std::string layer_A,std::string layer_B,std::string layer_C)
    {
        auto A = get_layer(layer_A);
        auto B = get_layer(layer_B);
        auto C = get_layer(layer_C);
        if (A->get_node_length()!=B->get_node_length())
        {
            //TODO:raise error
            throw "The operator's two layer 's length are not same! But the operator need same length";
        }else
        {
            int node_number = A->get_node_length();
            //std::shared_ptr<MathLayer> C = create_MathLayer(node_number, A->get_inner_function());
            std::vector<std::shared_ptr<Node> > input_node_A = A->get_node_list();
            std::vector<std::shared_ptr<Node> > input_node_B = B->get_node_list();
            std::vector<std::shared_ptr<Node> > node_list = C->get_node_list();
            for (int i=0;i<node_number;++i)
            {
                input_node_A[i]->connect(node_list[i],NodeFlow::OUTNODE);
                input_node_B[i]->connect(node_list[i],NodeFlow::OUTNODE);
                node_list[i]->connect(input_node_A[i],NodeFlow::INNODE);
                node_list[i]->connect(input_node_B[i],NodeFlow::INNODE);
            }
            return C;
        }
    }
    int get_param_number()
    {
        int ans=0;
        for (auto iter =layer_map_.begin();iter!=layer_map_.end();++iter )
        {
            ans += iter->second->get_param_number();
        }
        return ans;
    }
    void set_param_gradient(Eigen::MatrixXd* param,Eigen::MatrixXd* gradient)
    {
        int offset=0;
        for (auto iter = layer_map_.begin();iter!=layer_map_.end();++iter)
        {
            iter->second->set_param_gradient(param,gradient,offset);
        }
    }
    void initialize()
    {
        for (auto iter = layer_map_.begin();iter!=layer_map_.end();++iter)
        {
            iter->second->initialize();
        }
    }
    void solidify(int n,int asyn_index)
    {
        param_ = Eigen::MatrixXd(n,1);
        gradient_ = Eigen::MatrixXd(n,1);
        set_param_gradient(&param_,&gradient_);
        for (auto iter = layer_map_.begin();iter!=layer_map_.end();++iter)
        {
            iter->second->set_asyn_index(asyn_index);
        }
    }
    Eigen::MatrixXd get_param()
    {
        return param_;
    }
    void set_param(const Eigen::MatrixXd &param)
    {
        for (int i=0;i<param.rows();++i)
            param_(i,0) = param(i,0);
    }

};
}

#endif // MASTER_H_INCLUDED
