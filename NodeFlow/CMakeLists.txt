cmake_minimum_required(VERSION 2.8)

project(NodeFlow)

set(CMAKE_BUILD_TYPE "Release")

include_directories("./eigen3")

set(CMAKE_CXX_FLAGS "-std=c++11 -fopenmp -pthread")

add_executable(cnn sample_cnn.cpp)

add_executable(dense sample_dense.cpp)
#add_executable(BoundleAdjustmentByNode_accelebrate ba_main.cpp)

#target_link_libraries(bound_adjustment_ceres ${CERES_LIBRARIES})

#target_link_libraries(BoundleAdjustmentByNode_accelebrate ${CERES_LIBRARIES})
