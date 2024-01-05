#include "autograd/tensor.h"

// Constructor for scalar value
Tensor::Tensor(double value, bool requires_grad): requires_grad(requires_grad) {
    data = { {value} };
    shape = {1};
}

// Constructor for matrix.
Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<double>>> nestedList, bool requires_grad) : requires_grad(requires_grad) {
    extractShape(nestedList);
    extractData(nestedList);
}

// Overload << operator 
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    if (tensor.data.size() == 1 && tensor.data[0].size() == 1) {
        os << tensor.data[0][0]; // Scalar value
    } else {
        os << "shape=" << tensor.data.size() << "x" << tensor.data[0].size(); // Matrix shape
    }
    os << ", requires_grad=" << (tensor.requires_grad ? "true" : "false") << ")";
    return os;
}

// Recursive function for extracting shape from nested initializer list
template <typename T>
void Tensor::extractShape(const T& nestedList) {
    shape.push_back(nestedList.size());
    if (!nestedList.begin()->empty()) {
        extractShape(*nestedList.begin());
    }
}

// Recursive function for extracting data from nested initializer list
template <typename T>
void Tensor::extractData(const T& nestedList) {
    for (const auto& val : nestedList) {
        if (val.empty()) {
            extractData(*nestedList.begin());
        } else {
            data.push_back(static_cast<double>(val));
        }
    }
}

const std::vector<int>& Tensor::getShape() const {
    return shape;
}

const std::vector<std::vector<double>>& Tensor::getData() const {
    return data;
}

