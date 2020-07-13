# NodeFlow
Try to implement a neural net frame in C++ use the concept of node
<br> Because this project is composed of header files, it is easy to install. Put the file in any local address, and then include in CMakeLists. This project depends on the Eigen library. Since Eigen is also a header file, an Eigen library is provided directly in the project. This project provides two samples that illustrate the method of constructing MLP and CNN. sample_cnn.cpp reads MNIST and uses the classic LeNet-5 structure.
***
## HOW TO RUN:
<br>git clone https://github.com/starcosmos1225/NodeFlow.git
<br>cd NodeFlow/NodeFlow
<br>mkdir build && cd build
<br>cmake ..
<br>make -j4
<br>You will see two executable files: dense and cnn. Try to run it:
<br>./dense <"../Data/trainData.data"
<br>./cnn
