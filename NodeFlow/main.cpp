#include <unordered_map>
#include <iostream>
#include "eigen3/Eigen/Dense"
int main( )
   {
   using namespace std;
   unordered_map<int, Eigen::MatrixXd> v1, v2, v3;
   unordered_map<int, Eigen::MatrixXd>::iterator iter;
    Eigen::MatrixXd A(3,1);
    A<<1,2,3;
   v1.insert(pair<int,  Eigen::MatrixXd> (1, A));

   cout << "v1 = " ;
   for (iter = v1.begin(); iter != v1.end(); iter++)
      cout << iter->second << " ";
   cout << endl;

   v2 = v1;
   cout << "v2 = ";
   for (iter = v2.begin(); iter != v2.end(); iter++)
      cout << iter->second << " ";
   cout << endl;

// move v1 into v2
   v2.clear();
   v2 = move(v1);
   cout << "v2 = ";
   for (iter = v2.begin(); iter != v2.end(); iter++)
      cout << iter->second << " ";
   cout << endl;
   return 0;
   }
