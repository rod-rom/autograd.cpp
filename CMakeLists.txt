cmake_minimum_required(VERSION 3.12)
project(autograd++)

set(CMAKE_CXX_STANDARD 11)

include_directories(includes)

add_library(autograd++_lib
    src/autograd++/tensor.cpp)

add_executable(test_autograd++
    test/test_tensor.cpp)
target_link_libraries(test_autograd++ autograd++_lib)
