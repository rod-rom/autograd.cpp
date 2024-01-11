// tensor.h

#pragma once

#include <iostream>
#include <vector>
#include <initializer_list>
#include <type_traits>

class Tensor {
public:
    bool requires_grad;

    Tensor(double value, bool requires_grad = false);

    template <typename T, typename... Args>
    Tensor(const std::initializer_list<T>& values, Args... dims, bool requires_grad = false);

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    const std::vector<int>& getShape() const;
    const std::vector<double>& getData() const;

private:
    std::vector<double> data_;
    std::vector<int> dims_;
    
    template <typename T, typename... Args>
    void initData(const std::initializer_list<T>& values, int dim, Args... dims);
};
