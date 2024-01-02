#include "autograd/tensor.h"
#include <iostream>

int main() {
    // Test Tensor creation and printing
    Tensor scalar(42.0, true);
    std::cout << "Scalar: " << scalar << std::endl;

    Tensor matrix = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, true};
    std::cout << "Matrix: " << matrix << std::endl;

    // Add more test cases as needed

    return 0;
}
