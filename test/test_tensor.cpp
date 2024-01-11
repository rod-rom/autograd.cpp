#include "gtest/gtest.h"
#include "autograd/tensor.h"  

TEST(TensorTest, ScalarTensor) {
    Tensor scalar(3.14);
    EXPECT_EQ(scalar.getData()[0], 3.14);
    EXPECT_EQ(scalar.getShape(), std::vector<int>{1});
    std::stringstream ss;
    ss << scalar;
    EXPECT_EQ(ss.str(), "Tensor(3.14, requires_grad=false)");
}
/*
TEST(TensorTest, MatrixTensor) {
    Tensor matrix({ {1, 2, 3}, {4, 5, 6} });
    EXPECT_EQ(matrix.getData(), std::vector<double>{1., 2., 3., 4., 5., 6.});
    EXPECT_EQ(matrix.getShape(), std::vector<int>{2, 3});
    EXPECT_EQ(matrix.getData()[2], 3);  // Accessing an element
    std::stringstream ss;
    ss << matrix;
    EXPECT_EQ(ss.str(), "Tensor(shape=2x3, requires_grad=false)");
}

TEST(TensorTest, HigherDimensionalTensor) {
    Tensor tensor = { {{1, 2}, {3, 4}}, {{5, 6}, {7, 8}} };
    EXPECT_EQ(tensor.getShape(), std::vector<int>{2, 2, 2});
    EXPECT_EQ(tensor.getData()[4], 5);  // Accessing an element
    std::stringstream ss;
    ss << tensor;
    EXPECT_EQ(ss.str(), "Tensor(shape=2x2x2, requires_grad=false)");
}

TEST(TensorTest, GradientTracking) {
    Tensor tensor(2.5, true);  // Create with requires_grad=true
    EXPECT_TRUE(tensor.requires_grad);  // Check if gradient tracking is enabled
}
*/

