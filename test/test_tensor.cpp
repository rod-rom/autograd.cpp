#include "gtest/gtest.h"
#include "autograd/tensor.h"  

TEST(TensorTest, ScalarConstructor) {
    Tensor tensor(42.0, true);
    ASSERT_EQ(tensor.getShape().size(), 1);
    ASSERT_EQ(tensor.getShape()[0], 1);
    ASSERT_EQ(tensor.getData().size(), 1);
    ASSERT_EQ(tensor.getData()[0][0], 42.0);
}

TEST(TensorTest, MatrixConstructor) {
    Tensor matrix({ {{1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}} }, true);
    ASSERT_EQ(matrix.getShape().size(), 2);
    ASSERT_EQ(matrix.getShape()[0], 3);
    ASSERT_EQ(matrix.getShape()[1], 3);
    ASSERT_EQ(matrix.getData().size(), 3);
    ASSERT_EQ(matrix.getData()[0][0], 1.0);
    ASSERT_EQ(matrix.getData()[2][2], 9.0);
}

TEST(TensorTest, FourDimMatrixConstructor) {
    Tensor fourDimMatrix(
        {
            {
                {
                    { {1., 2.}, {3., 4.} },
                    { {5., 6.}, {7., 8.} }
                },
                {
                    { {9., 10.}, {11., 12.} },
                    { {13., 14.}, {15., 16.} }
                }
            }
        },
        true
    );

    ASSERT_EQ(fourDimMatrix.getShape().size(), 4);
    ASSERT_EQ(fourDimMatrix.getShape()[0], 2);
    ASSERT_EQ(fourDimMatrix.getShape()[1], 2);
    ASSERT_EQ(fourDimMatrix.getShape()[2], 2);
    ASSERT_EQ(fourDimMatrix.getShape()[3], 2);
    ASSERT_EQ(fourDimMatrix.getData().size(), 2);
    ASSERT_EQ(fourDimMatrix.getData()[0][0][0][0], 1.0);
    ASSERT_EQ(fourDimMatrix.getData()[1][1][1][1], 16.0);
}



