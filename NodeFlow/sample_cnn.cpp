#define _BSD_SOURCE
#include <sys/time.h>
#include<bits/stdc++.h>
#include "NodeFlow.h"

class CNN
{
private:
    int output_;
    NodeFlow::Session nf;
    std::string minimizelayer_name_;
public:
    CNN(int output=10):output_(output)
    {
        nf = NodeFlow::Session(1,1);
        try
        {
            auto input_layer = nf.create_InputLayer(28*28,"input",std::vector<int>{28,28});
            //input_layer->set_shape(std::vector<int>{28,28});
            std::vector<int> cnn_shape{5,5,6};
            NodeFlow::Glorot init(28*28,10);
            auto l2 = nf.create_CNNLayer(input_layer, cnn_shape,init,
                                        NodeFlow::NORMAL_FUNCTION, NodeFlow::TANH, NodeFlow::ADAM,"l2");

            std::vector<int> pooling_size{2,2};
            auto l2p = nf.create_poolingLayer(l2,pooling_size,2,
                                            NodeFlow::MAX_FUNCTION,"l2p");
            std::vector<int> cnn_shape1{5,5,16};
            auto l3 = nf.create_CNNLayer(l2p,  cnn_shape1,init,
                                        NodeFlow::NORMAL_FUNCTION, NodeFlow::TANH, NodeFlow::ADAM,"l3");
            std::vector<int> pooling_size1{2,2};
            auto l3p = nf.create_poolingLayer(l3,pooling_size1,2,
                                            NodeFlow::MAX_FUNCTION,"l3p");
            std::vector<int> cnn_shape2{4,4,120};
            auto l4 = nf.create_CNNLayer(l3p,cnn_shape2,init,NodeFlow::NORMAL_FUNCTION,NodeFlow::TANH,
                                             NodeFlow::ADAM,"l4");
            auto f4 = nf.create_DenseLayer(l4, 84, init,NodeFlow::NORMAL_FUNCTION, NodeFlow::TANH, NodeFlow::ADAM,"f4");
            auto output_layer = nf.create_DenseLayer(f4, 10, init,NodeFlow::NORMAL_FUNCTION, NodeFlow::NONE, NodeFlow::ADAM,"output");
            //nf.get_optimizer(NodeFlow::GRADIENTDESCENT)->set_option(0.00001);
            auto label_layer = nf.create_InputLayer(output,"label");
            auto minimizelayer = nf.square(nf.minus(output_layer,label_layer));
            minimizelayer_name_ = minimizelayer->get_name();
            //std::cout<<minimizelayer_name_<<std::endl;
            nf.solidify();
            //version3.0 layer has its scale to change the learning rate;

        }catch(const char* err)
        {
            std::cout<<"Error:"<<err<<std::endl;
            exit(0);
        }


    }
    void learn(std::vector<std::vector<double>> data,std::vector<std::vector<double >> label)
    {
        std::unordered_map<std::string, std::vector<std::vector<double> > > dict;
        dict["input"] = data;
        dict["label"] =  label;
        std::vector<std::vector<double> > loss = nf.optimize(minimizelayer_name_,dict,NodeFlow::SGD);
        double loss_=0.0;
        int count = 0;
        for (auto line:loss)
        {
            for (auto l:line)
            {
                loss_ +=l;
                count++;
            }

        }
        std::cout<<std::setprecision(16)<<"loss:"<<loss_*1.0/count<<std::endl;
    }
    std::vector<std::vector<double> > predict(std::vector<std::vector<double> > data)
    {
        std::unordered_map<std::string, std::vector<std::vector<double> > > dict;
        dict["input"] = data;
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
void readMNIST(std::vector<std::vector<double> > &data,std::vector<std::vector<double> > &label,
               std::vector<std::vector<double> > &testdata,std::vector<std::vector<double> > &testlabel)
{
	std::fstream inFile;
    inFile.open("../Data/train-labels.idx1-ubyte",std::ios::binary|std::ios::in);
    char buffer[4];
    char l[1];
    inFile.read(buffer,4);
    inFile.read(buffer,4);
    int number = (unsigned char)(buffer[0])*256*256*256+(unsigned char)(buffer[1])*256*256+(unsigned char)(buffer[2])*256+(unsigned char)(buffer[3]);
    for (int i=0;i<number;i++)
    {
        inFile.read(l,1);
        label.push_back(NodeFlow::onehot(l[0],10));

       //label.push_back(std::vector<double>{((unsigned char)l[0])*1.0});
    }
    inFile.close();
    inFile.open("../Data/train-images.idx3-ubyte",std::ios::binary|std::ios::in);
    char l1[784];
    inFile.read(buffer,4);
    inFile.read(buffer,4);
    number = (unsigned char)(buffer[0])*256*256*256+(unsigned char)(buffer[1])*256*256+(unsigned char)(buffer[2])*256+(unsigned char)(buffer[3]);
    inFile.read(buffer,4);
    inFile.read(buffer,4);
    for (int i=0;i<number;i++)
    {
        std::vector<double > la(784);
        inFile.read(l1,784);
        for (int i=0;i<784;i++)
        {
            la[i] = ((((unsigned char)l1[i])*1.0)-127.5)/127.5;
        }
        data.push_back(la);
    }
    inFile.close();
    inFile.open("../Data/t10k-labels.idx1-ubyte",std::ios::binary|std::ios::in);
    inFile.read(buffer,4);
    inFile.read(buffer,4);
    number = (unsigned char)(buffer[0])*256*256*256+(unsigned char)(buffer[1])*256*256+(unsigned char)(buffer[2])*256+(unsigned char)(buffer[3]);
    for (int i=0;i<number;i++)
    {
        inFile.read(l,1);
        testlabel.push_back(NodeFlow::onehot(l[0],10));

       //label.push_back(std::vector<double>{((unsigned char)l[0])*1.0});
    }
    inFile.close();
    inFile.open("../Data/t10k-images.idx3-ubyte",std::ios::binary|std::ios::in);
    inFile.read(buffer,4);
    inFile.read(buffer,4);
    number = (unsigned char)(buffer[0])*256*256*256+(unsigned char)(buffer[1])*256*256+(unsigned char)(buffer[2])*256+(unsigned char)(buffer[3]);
    inFile.read(buffer,4);
    inFile.read(buffer,4);
    for (int i=0;i<number;i++)
    {
        std::vector<double > la(784);
        inFile.read(l1,784);
        for (int i=0;i<784;i++)
        {
            la[i] = ((((unsigned char)l1[i])*1.0)-127.5)/127.5;
        }
        testdata.push_back(la);
    }
    inFile.close();
}

int main()
{

    CNN net(10);
    std::vector<std::vector<double> > testdata;
    std::vector<std::vector<double> > testlabel;
    std::vector<std::vector<double> > trainData;
    std::vector<std::vector<double> > trainLabel;
    std::vector<std::vector<double> > traindata;
    std::vector<std::vector<double> > trainlabel;
    readMNIST(trainData,trainLabel,testdata,testlabel);
    for (unsigned int i=0;i<1;++i)
    {
        traindata.push_back(trainData[i]);
        trainlabel.push_back(trainLabel[i]);
    }
    std::cout<<"begin train ADAM test time"<<std::endl;
    for (int i=0;i<500;i++)
    {
        net.learn(trainData,trainLabel);
        std::vector<std::vector<double> > result = net.predict(testdata);
        int right_rate=0;
        for (int j=0;j<testlabel.size();j++)
        {

            if (testlabel[j][NodeFlow::argmax(result[j])]==1.0)
                right_rate++;
        }
        std::cout<<"the right rate:"<<right_rate*1.0/testlabel.size()<<std::endl;
    }
    return 0;
}
