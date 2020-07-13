#pragma once
#include <bits/stdc++.h>

namespace NodeFlow
{
struct subNode
{
    std::string name_;
    std::vector<std::shared_ptr<NodeFlow::subNode> > next;
    subNode(){}
    subNode(std::string name):name_(name){}
};
class Graph
{
private:
    std::unordered_map<std::string, std::shared_ptr<NodeFlow::subNode> > graph_;
    int node_number;
public:
    Graph(){node_number=0;}
    //get the graph
    std::unordered_map<std::string, std::shared_ptr<NodeFlow::subNode> > get_graph()
    {
        return graph_;
    }

    //regist a new layer's name to the graph
    void regist_layer(std::string name)
    {
        if (graph_.find(name)==graph_.end())
        {
            graph_[name] = std::make_shared<NodeFlow::subNode>(name);
            node_number ++;
        }else
        {
            //TODO raise a error
            throw "the layer is alread in the graph";
        }
    }
    int get_node_number(){return node_number;}
    //build a link from name1 to name2
    void build_link(std::string name1,std::string name2)
    {
        graph_[name1]->next.push_back(graph_[name2]);
    }
    std::vector<std::string> get_topology(std::string name)
    {
        //the post-order of the tree is the topology order of the tree
        std::vector<std::string > topo;
        //avoid repeat put
        std::set<std::string> blank;
        post_order(name,topo,blank);
        return topo;
    }
    void post_order(std::string name, std::vector<std::string> &topo,std::set<std::string> &blank)
    {
        for (auto n:graph_[name]->next)
        {
            post_order(n->name_,topo,blank);
        }
        if (blank.find(name)==blank.end())
        {
            topo.push_back(name);
            blank.insert(name);
        }
    }
    //version 3.0 compute the level of layer from some layer
    std::unordered_map<std::string, int> compute_layer_level(std::string layer)
    {
        std::queue<std::string > q;
        std::unordered_map<std::string, int> level;
        level[layer] = 0;
        q.push(layer);
        while (!q.empty())
        {
            std::string tmp = q.front();
            q.pop();
            if (graph_.find(tmp)!=graph_.end())
            {
                for(auto next:graph_[tmp]->next)
                {
                    level[next->name_] = level[tmp]+1;
                    q.push(next->name_);
                }
            }
        }
        return level;
    }
    bool find(std::string layer_name)
    {
        if (graph_.find(layer_name)!=graph_.end())
            return true;
        return false;
    }

};
}


