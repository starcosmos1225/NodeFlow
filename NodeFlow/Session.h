#ifndef SESSION_H_INCLUDED
#define SESSION_H_INCLUDED
#include "Layer.h"
#include "activation_function.h"
#include "optimizer.h"
#include "Graph.h"
#include "Master.h"
#include "util.h"
#include<bits/stdc++.h>
#include "omp.h"
namespace NodeFlow
{

class Session
{
private:
    std::vector<NodeFlow::Master> masters;
    std::unordered_map<int, std::shared_ptr<NodeFlow::optimizer> > Optimizer_list;
    NodeFlow::Graph graph_;
    int thread_number_;
    int batch_size_;
    //version3.0 param and gradient become global variable
    Eigen::MatrixXd param_;
    Eigen::MatrixXd gradient_;
public:
    Session(int thread_number=4,int batch_size=64)
    {
        //optimizer:GradientDescent,RMSPRop,
        Optimizer_list[NodeFlow::RMSPROP] = std::make_shared<NodeFlow::RMSProp>();
        Optimizer_list[NodeFlow::GRADIENTDESCENT] = std::make_shared<NodeFlow::GradientDescent>();
        Optimizer_list[NodeFlow::MOMENTUM] = std::make_shared<NodeFlow::MomentumOptimizer>();
        Optimizer_list[NodeFlow::ADAM] = std::make_shared<NodeFlow::Adam>();
        Adam_iteration = 0;
        thread_number_ = thread_number;
        for (int i=0;i<thread_number_;++i)
        {
            masters.push_back(NodeFlow::Master());
        }
        batch_size_ = batch_size;
    }
    std::shared_ptr<NodeFlow::optimizer> get_optimizer(int optimizer_type)
    {
        if (Optimizer_list.find(optimizer_type)==Optimizer_list.end())
        {
            throw "No such optimizer";
        }else
        {
            return Optimizer_list[optimizer_type];
        }
    }
    void regist_layer(std::string name)
    {
        if (graph_.find(name)==false)
        {
            graph_.regist_layer(name);
        }else
        {
            //TODO raise a error
            throw "The layer is already in the Graph";
        }
    }
    std::string get_random_name()
    {
        std::string name = "nf";
        int length = graph_.get_node_number();
        std::string new_name =name + std::to_string(length);
        while (graph_.find(new_name))
        {
            length++;
            new_name =name + std::to_string(length);
        }
        return new_name;
    }
    //build a link from layer1 to layer2. But in the graph it is reverse from layer2 to layer1 because the graph we build is a reverse-tree
    void build_link(std::string layer1,std::string layer2)
    {
        graph_.build_link(layer2,layer1);
    }
    void minimize(std::string layer)
    {
        std::vector<std::string> layer_list = graph_.get_topology(layer);
        for (int i=layer_list.size()-1;i>=0;--i)
        {
            masters[0].get_layer(layer_list[i])->update();
        }
        gradient_.setZero();
    }
    //optimize = compute(layer)+minize(layer)
    std::vector<std::vector<double> > evaluate(std::string layer,
                                              std::unordered_map<std::string,std::vector< std::vector<double> > > &dict)
    {
        std::vector<std::string> layer_list = graph_.get_topology(layer);
        gradient_.setZero();
        return masters[0].compute(layer_list,layer,dict,true);
    }
    void thread_optimize(int index,
                        std::string layer,
                        std::vector<std::string> &layer_list,
                        std::unordered_map<std::string,std::vector< std::vector<double> > > &dict,
                        std::vector<std::vector<double> > &result,
                        std::unordered_map<std::string,std::vector< std::vector<double> > > &test_dict)
    {
        //std::cout<<"compute1:"<<"index:"<<index<<std::endl;
        masters[index].compute(layer_list,layer,dict);
        //std::cout<<"compute2:"<<"index:"<<index<<std::endl;
        masters[index].minimize(layer,layer_list);
        //std::cout<<"compute3:"<<"index:"<<index<<std::endl;
        result =  masters[index].compute(layer_list,layer,test_dict);
    }
    void thread_compute(int index,
                        std::string layer,
                        std::vector<std::string> &layer_list,
                        std::unordered_map<std::string,std::vector< std::vector<double> > > &dict,
                        std::vector<std::vector<double> > &result)
    {
        result = masters[index].compute(layer_list,layer,dict);

    }
    std::vector<std::vector<double> > optimize(
                                               std::string layer,
                                               std::unordered_map<std::string,std::vector< std::vector<double> > > &dict,
                                               int optimize_type=NodeFlow::NONE)
    {
        //compute_layer_scale(layer);
        NodeFlow::Adam_iteration++;
        std::vector<std::string> layer_list = graph_.get_topology(layer);
        if (optimize_type==NodeFlow::SGD_MULTIPLE)
        {
            //TODO
            //first get thread_number_*batch_size_ random data
            //put each batch_size_data to the masters
            //optimize each one
            //use a random batch_size_ to check the best result
            //copy the masters param to other one
             int datalength = dict.begin()->second.size();
             std::vector<int> random_order = NodeFlow::create_random_int((thread_number_+1)*batch_size_,datalength);
             std::vector<std::unordered_map<std::string,std::vector< std::vector<double> > >> thread_dict(thread_number_+1);
            std::vector<std::vector<std::vector<double> >> loss(thread_number_);
            int index = 0;
            for (int i=0;i<=thread_number_;++i)
            {
                for (int j=0;j<batch_size_;++j)
                {
                    for (auto iter = dict.begin();iter!=dict.end();iter++)
                    {
                        std::string layer_name = iter->first;
                        thread_dict[i][layer_name].push_back((iter->second)[random_order[index]]);
                    }
                    index++;
                }
            }
            //std::cout<<"begin thread"<<std::endl;
            std::vector<std::thread> thread_(thread_number_);
            for (int i=0;i!=thread_number_;++i)
            {
                thread_[i] = std::thread(&NodeFlow::Session::thread_optimize,this,i,layer,std::ref(layer_list),std::ref(thread_dict[i]),
                                  std::ref(loss[i]),std::ref(thread_dict[thread_number_]));
            }
            for(int i=0;i!=thread_number_;++i)
            {
                thread_[i] .join();
            }
            //std::cout<<"end thread"<<std::endl;
            int max_i;
            double min_loss = DBL_MAX;
            for (int i=0;i<thread_number_;++i)
            {
                double loss_=0.0;
                for (auto &l:loss[i])
                {
                    loss_ +=l[0];
                }
                if (loss_<min_loss)
                {
                    max_i = i;
                    min_loss=loss_;
                }
            }
            for (int i=0;i<thread_number_;++i)
            {
                if (i==max_i)
                    continue;
                masters[i].set_param(masters[max_i].get_param());
                RMS_R_MAP[i] = RMS_R_MAP[max_i];
                ADAM_V_MAP[i] =ADAM_V_MAP[max_i]  ;
                ADAM_M_MAP[i] =ADAM_M_MAP[max_i]  ;
                MOMENTUM_MAP[i] = MOMENTUM_MAP[max_i];
            }
            //std::cout<<"max master:"<<max_i<<std::endl;
            std::vector<std::vector<double> > ans;
            for (int i=0;i!=thread_number_;++i)
            {
                 ans.insert(ans.end(), loss[i].begin(), loss[i].end());
            }
            return ans;
        }else if(optimize_type==NodeFlow::SGD)
        {
            if (NodeFlow::Adam_iteration>1)
            {
                masters[0].get_layer("output")->stop_gradient();
            }
            if (NodeFlow::Adam_iteration>6)
            {
                masters[0].get_layer("f4")->stop_gradient();
            }
            gradient_.setZero();
            int datalength = dict.begin()->second.size();
            std::vector<int> random_order = NodeFlow::create_random_int(datalength/10*9,datalength);
            std::vector<std::vector<double> > loss;
            double total_l=0.0;
            int right_rate = 0;
            for (int i=0;i<random_order.size();i+=batch_size_)
            {
                //get one dict data
                std::unordered_map<std::string,std::vector< std::vector<double> > > dict_one;
                for (auto iter = dict.begin();iter!=dict.end();iter++)
                {
                    std::string layer_name = iter->first;
                    for (int j=0;j<batch_size_;++j)
                        dict_one[layer_name].push_back((iter->second)[random_order[i+j]]);
                        //dict_one[layer_name].push_back((iter->second)[i]);
                }
                //optimize
                std::vector<std::vector<double>> loss_ = masters[0].compute(layer_list,layer,dict_one);
                int max_index = NodeFlow::argmax(masters[0].get_layer("output")->get_value());
                std::vector<double> single_label = dict["label"][random_order[i]];
                if (fabs(single_label[max_index]-1.0)<0.000001)
                {
                    right_rate++;
                }
                double total_a = 0.0;
                int count=0;
                for (std::vector<double> &l:loss_)
                {
                    loss.push_back(l);

                    for (double a:l)
                    {
                        total_a +=a;
                        count ++;
                    }
                }
                total_l = (total_l*i+total_a)/(i+1);
                std::cout<<"index:"<<i<<" loss:"<<total_l*1.0/count<<
                           " rate:"<<right_rate*1.0/(i+1)<<'\r';

                //system("clear");
                minimize(layer);
            }
            return loss;
        }else
        {
            //TODO thread
            std::vector<std::vector<std::vector<double> > > loss(thread_number_);
            std::vector<std::unordered_map<std::string,std::vector< std::vector<double> > > > thread_dict(thread_number_);
            int data_length = dict.begin()->second.size();
            int split = data_length/thread_number_;
            for (int i=0;i!=thread_number_;++i)
            {
                for (auto iter=dict.begin();iter!=dict.end();++iter)
                {
                    thread_dict[i][iter->first] = std::vector< std::vector<double> >(iter->second.begin()+i*split,iter->second.begin()+(i+1)*split);
                }
            }
            std::vector<std::thread> thread_(thread_number_);
            for (int i=0;i!=thread_number_;++i)
            {
                thread_[i] = std::thread(&NodeFlow::Session::thread_compute,this,i,layer,std::ref(layer_list),std::ref(thread_dict[i]),
                                  std::ref(loss[i]));
            }
            for(int i=0;i!=thread_number_;++i)
            {
                thread_[i] .join();
            }
            std::vector<std::vector<double> > loss_;
            for (int i=0;i!=thread_number_;++i)
            {
                loss_.insert(loss_.end(), loss[i].begin(), loss[i].end());
            }
            minimize(layer);
            return loss_;
        }
    }
    //get the layer's name through the layer's point
    std::string get_layer_name(std::shared_ptr<Layer> layer)
    {

        return masters[0].get_layer_name(layer);
    }
    /*create Layers*/
    std::shared_ptr<Layer> create_DenseLayer(std::shared_ptr<Layer> input ,
               int node_number,
               NodeFlow::initializer &init,
               int inner_function_name=NodeFlow::NORMAL_FUNCTION,//inner_function
               int activation_function_name=NodeFlow::NONE,//nullptr
               int optimizer_function_name=NodeFlow::RMSPROP,//RMSProp
               std::string name = NodeFlow::NONENAME)
    {
        std::string layer_name = input->get_name();
        std::string name_;
        if (name == NodeFlow::NONENAME)
        {
            name_ = get_random_name();
        }else
        {
            name_ = name;
        }
        for (auto &master:masters)
        {
            master.create_DenseLayer(layer_name,node_number,init,inner_function_name,
                                     activation_function_name,
                                     Optimizer_list[optimizer_function_name],name_);
        }
        regist_layer(name_);
        build_link(layer_name,name_);
        return masters[0].get_layer(name_);

    }
    std::shared_ptr<Layer> create_SparseLayer(std::shared_ptr<Layer> input ,
                                              int size,
                                              int step,
                                              NodeFlow::initializer &init,
                                              int inner_function_name=NodeFlow::NORMAL_FUNCTION,//inner_function
                                              int activation_function_name=NodeFlow::NONE,//nullptr
                                              int optimizer_function_name=NodeFlow::RMSPROP,//RMSProp
                                              std::string name = NodeFlow::NONENAME)
    {
        std::string layer_name = input->get_name();
        std::string name_;
        if (name == NodeFlow::NONENAME)
        {
            name_ = get_random_name();
        }else
        {
            name_ = name;
        }
        for (auto &master:masters)
        {
            master.create_SparseLayer(layer_name,size,step,init,inner_function_name,
                                     activation_function_name,
                                     Optimizer_list[optimizer_function_name],name_);
        }
        regist_layer(name_);
        build_link(layer_name,name_);
        return masters[0].get_layer(name_);

    }
    std::shared_ptr<Layer> create_CNNLayer(
            std::shared_ptr<Layer> input ,
            std::vector<int> cnn_shape,
            NodeFlow::initializer &init,
            int inner_function_name=NodeFlow::NORMAL_FUNCTION,//inner_function
            int activation_function_name=NodeFlow::NONE,//nullptr
            int optimizer_function_name=NodeFlow::RMSPROP,//RMSProp
            std::string name = NodeFlow::NONENAME)
    {
        std::string layer_name = input->get_name();
        std::string name_;
        if (name == NodeFlow::NONENAME)
        {
            name_ = get_random_name();
        }else
        {
            name_ = name;
        }
        for (auto &master:masters)
        {
            master.create_CNNLayer(layer_name,cnn_shape,init,inner_function_name,activation_function_name,
                                     Optimizer_list[optimizer_function_name],name_);
        }
        regist_layer(name_);
        build_link(layer_name,name_);
        return masters[0].get_layer(name_);

    }
    std::shared_ptr<Layer> create_poolingLayer(
            std::shared_ptr<Layer> input ,
            std::vector<int> pooling_shape,
            int step_size=2,
            int inner_function_name=NodeFlow::MAX_FUNCTION,//inner_function
            std::string name = NodeFlow::NONENAME)
    {
        std::string layer_name = input->get_name();
        std::string name_;
        if (name == NodeFlow::NONENAME)
        {
            name_ = get_random_name();
        }else
        {
            name_ = name;
        }
        for (auto &master:masters)
        {
            master.create_poolingLayer(layer_name,pooling_shape,step_size,inner_function_name,name_);
        }
        regist_layer(name_);
        build_link(layer_name,name_);
        return masters[0].get_layer(name_);

    }
    std::shared_ptr<Layer> create_InputLayer(int node_number,
                                             std::string name=NodeFlow::NONENAME,
                                             std::vector<int> shape=std::vector<int>(0))
    {
        std::string name_ = name;
        for (auto &master:masters)
        {
            master.create_InputLayer(node_number,shape,name_);
        }
        regist_layer(name_);
        //masters[0].get_layer(name_)->get_name();
        //masters[0].get_layer(name_)->set_shape(std::vector<int>{28,28});
        return masters[0].get_layer(name_);
    }
    std::shared_ptr<Layer> create_MathLayer(int node_number,
                                             int f=NodeFlow::NORMAL_FUNCTION,
                                            int activation_function_name = NodeFlow::NONE,
                                            std::shared_ptr<NodeFlow::optimizer> opt=nullptr,
                                            std::string name = NodeFlow::NONENAME)
    {
        std::string name_;
        if (name == NodeFlow::NONENAME)
        {
            name_ = get_random_name();
        }else
        {
            name_ = name;
        }
        for (auto &master:masters)
        {
            master.create_MathLayer(node_number,f,activation_function_name,opt,name_);
        }
        regist_layer(name_);
        return masters[0].get_layer(name_);
    }
    /*MathLayer add minus square*/
    std::shared_ptr<Layer> add(std::shared_ptr<Layer> A,std::shared_ptr<Layer> B)
    {
        if (A->get_node_length()!=B->get_node_length())
        {
            //TODO:raise error
            throw "The operator's two layer 's length are not same! But the operator need same length";
        }else
        {
            int node_number = A->get_node_length();
            std::string name_A = A->get_name();
            std::string name_B = B->get_name();
            auto C = create_MathLayer(node_number, NodeFlow::ADD_FUNCTION);
            for (auto &master:masters)
            {
                master.add(name_A,name_B,C->get_name());
            }
            build_link(A->get_name(),C->get_name());
            build_link(B->get_name(),C->get_name());
            return C;
        }
    }
    std::shared_ptr<Layer> minus(std::shared_ptr<Layer> A,std::shared_ptr<Layer> B)
    {
        if (A->get_node_length()!=B->get_node_length())
        {
            //TODO:raise error
            throw "The operator's two layer 's length are not same! But the operator need same length";
        }else
        {
            int node_number = A->get_node_length();
            std::string name_A = A->get_name();
            std::string name_B = B->get_name();
            auto C = create_MathLayer(node_number, NodeFlow::MINUS_FUNCTION);
            for (auto &master:masters)
            {
                master.minus(name_A,name_B,C->get_name());
            }
            build_link(A->get_name(),C->get_name());
            build_link(B->get_name(),C->get_name());
            return C;
        }
    }
    std::shared_ptr<Layer> square(std::shared_ptr<Layer> A)
    {
        std::string name_ = A->get_name();
        for (auto &master:masters)
        {
            auto layer = master.get_layer(name_);
            layer->add_activation_function(std::make_shared<NodeFlow::activation::square>());
        }
        return masters[0].get_layer(name_);
    }
    //version 3.0 compute layer's scale for learning rate
    void compute_layer_scale(std::string start_layer)
    {
        auto level = graph_.compute_layer_level(start_layer);
        for (auto iter = level.begin();iter!=level.end();iter++)
        {
            for (auto &master:masters)
            {
                master.get_layer(iter->first)->set_scale(pow(NodeFlow::LAYER_SCALE,iter->second));
            }
        }
    }
    void solidify()
    {
        int n = masters[0].get_param_number();
        std::cout<<"number:"<<n<<std::endl;
        param_ = Eigen::MatrixXd(n,1);
        gradient_ = Eigen::MatrixXd(n,1);
        for (int i=0;i<thread_number_;++i)
        {
            masters[i].set_param_gradient(&param_,&gradient_);

        }
        masters[0].initialize();
    }
    void asyn_solidify()
    {
        int n = masters[0].get_param_number();
        for (int i=0;i<thread_number_;++i)
        {
            masters[i].solidify(n,i);
            masters[i].initialize();
        }

    }
    //version 4.0 save
    void save(std::string filename)
    {
        std::ofstream savefile(filename);
        std::unordered_map<std::string, std::shared_ptr<NodeFlow::subNode> > graph = graph_.get_graph();
        for (auto iter = graph.begin();iter!=graph.end();++iter)
        {
            auto layer = masters[0].get_layer(iter->first);
            layer->save(savefile);
        }
        savefile.close();
    }
    void load(std::string filename)
    {
        std::ifstream loadfile(filename);
        if (!loadfile.is_open())
        {
            throw "load file fail!";
        }else
        {
            std::string layer_name;
            int node_number;
            loadfile>>layer_name>>node_number;
            if (graph_.find(layer_name))
            {
                auto layer = masters[0].get_layer(layer_name);
                if (layer->get_param_number()==node_number)
                {
                    layer->load(loadfile,node_number);
                }else
                {
                    throw "different load layer number!";
                }
            }else
            {
                throw "could not load this layer because the layer name do not exist!";
            }
        }


    }
    void display(std::string layer)
    {
        masters[0].get_layer(layer)->display();
    }
};
}

#endif // MASTER_H_INCLUDED

