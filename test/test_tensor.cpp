#include "gtest/gtest.h"
#include "autograd/tensor.h"  

TEST(TensorTest, ScalarConstructor) {
    Tensor tensor(42.0, true);
    ASSERT_EQ(tensor.shape.size(), 1);
    ASSERT_EQ(tensor.shape[0], 1);
    ASSERT_EQ(tensor.data.size(), 1);
    ASSERT_EQ(tensor.data[0][0], 42.0);
}

TEST(TensorTest, MatrixConstructor) {
    Tensor matrix({ {1, 2, 3}, {4, 5, 6}, {7, 8, 9} }, true);
    ASSERT_EQ(matrix.shape.size(), 2);
    ASSERT_EQ(matrix.shape[0], 3);
    ASSERT_EQ(matrix.shape[1], 3);
    ASSERT_EQ(matrix.data.size(), 3);
    ASSERT_EQ(matrix.data[0][0], 1.0);
    ASSERT_EQ(matrix.data[2][2], 9.0);
}

TEST(TensorTest, FourDimMatrixConstructor) {
    Tensor fourDimMatrix(
        {
            {
                {{1, 2}, {3, 4}},
                {{5, 6}, {7, 8}}
            },
            {
                {{9, 10}, {11, 12}},
                {{13, 14}, {15, 16}}
            }
        },
        true
    );

    ASSERT_EQ(fourDimMatrix.shape.size(), 4);
    ASSERT_EQ(fourDimMatrix.shape[0], 2);
    ASSERT_EQ(fourDimMatrix.shape[1], 2);
    ASSERT_EQ(fourDimMatrix.shape[2], 2);
    ASSERT_EQ(fourDimMatrix.shape[3], 2);
    ASSERT_EQ(fourDimMatrix.data.size(), 2);
    ASSERT_EQ(fourDimMatrix.data[0][0][0][0], 1.0);
    ASSERT_EQ(fourDimMatrix.data[1][1][1][1], 16.0);
}


