#pragma once
#include<bits/stdc++.h>
#include "test2.h"
class B;
class A
{
public:
    A(){}
    void useB(B b)
    {
        b.display;
    }
    void display()
    {
        std::cout<<"call A"<<std::endl;
    }
};
