#pragma once
#include<bits/stdc++.h>
#include "test.h"
class B
{
public:
    B(){}
    void useA(A a)
    {
        A.display();
    }
    void display()
    {
        std::cout<<"B call"<<std::endl;
    }
};
