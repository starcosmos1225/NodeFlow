#include<bits/stdc++.h>
#include "NodeFlow.h"

class NN
{
private:
    int input_;
    int output_;
    NodeFlow::Session nf;
    std::string minimizelayer_name_;
public:
    NN(int input = 4,int output=1):input_(input),output_(output)
    {
        nf = NodeFlow::Session();
        auto input_layer = nf.create_InputLayer(4,"input");
        std::cout<<input_layer->get_name()<<std::endl;
        NodeFlow::Guass init(0,0.1);
        auto l2 = nf.create_DenseLayer(input_layer, 20, init,NodeFlow::NORMAL_FUNCTION, NodeFlow::SIGMOID, NodeFlow::RMSPROP,"l2");
        std::cout<<l2->get_name()<<std::endl;
        //std::cout<<"node number:"<<l2->get_node_length()<<std::endl;
        //std::cout<<"node_number:"<<l2->get_node_list().size()<<std::endl;
        auto output_layer = nf.create_DenseLayer(l2, 1, init,NodeFlow::NORMAL_FUNCTION, NodeFlow::NONE, NodeFlow::RMSPROP,"output");
        std::cout<<output_layer->get_name()<<std::endl;
        auto label_layer = nf.create_InputLayer(output,"label");
        std::cout<<label_layer->get_name()<<std::endl;
        auto minimizelayer = nf.square(nf.minus(output_layer,label_layer));
        minimizelayer_name_ = minimizelayer->get_name();
        nf.solidify();
        //std::cout<<"node number:"<<minimizelayer->get_node_list().size()<<std::endl;

        std::cout<<"minimizename:"<<minimizelayer_name_<<std::endl;
    }
    void learn(std::vector<std::vector<double>> data,std::vector<std::vector<double >> label)
    {
        std::unordered_map<std::string, std::vector<std::vector<double> > > dict;
        dict["input"] = data;
        dict["label"] =  label;
        std::vector<std::vector<double> > loss = nf.optimize(minimizelayer_name_,dict);
        double loss_=0.0;
        for (auto l:loss)
        {
            loss_ +=l[0];
        }
        std::cout<<std::setprecision(16)<<"loss:"<<loss_*0.5/900<<std::endl;
    }
    std::vector<std::vector<double> > predict(std::vector<std::vector<double> > data)
    {
        std::unordered_map<std::string, std::vector<std::vector<double> > > dict;
        dict["input"] = data;
        //nf->stop_all_gradient();
        return nf.evaluate("output", dict);
    }
};
void readData(std::vector<std::vector<double> > &data,std::vector<std::vector<double> > &label)
{
    int n,m;
    std::cin>>n>>m;
    for (int i=0;i<n;++i)
    {
        std::vector<double> d;
        for (int j=0;j<m;++j)
        {
            double number;
            std::cin>>number;
            d.push_back(number);
        }
        data.push_back(d);
        double number;
        std::cin>>number;
        std::vector<double> l;
        l.push_back(number);
        label.push_back(l);
    }
}

int main()
{
    NN net(4,1);
    std::vector<std::vector<double> > data;
    std::vector<std::vector<double> > label;
    std::vector<std::vector<double> > trainData;
    std::vector<std::vector<double> > trainLabel;
    std::vector<std::vector<double> > testData;
    std::vector<std::vector<double> > testLabel;
    readData(data,label);
    for (unsigned int i=0;i<int(data.size()*9/10);++i)
    {
        trainData.push_back(data[i]);
        trainLabel.push_back(label[i]);
    }
    for (unsigned int i=int(data.size()*9/10);i<data.size();++i)
    {
        testData.push_back(data[i]);
        testLabel.push_back(label[i]);
    }
    for (int i=0;i<500;i++)
    {
        net.learn(trainData,trainLabel);

    }
    return 0;
}
