#include "autograd/tensor.h"

// Constructor for scalar value
Tensor::Tensor(double value, bool requires_grad): requires_grad(requires_grad) {
    data_ = {value};
    dims_ = {1};
}

// Constructor for matrix.
template <typename T, typename... Args>
Tensor::Tensor(const std::initializer_list<T>& values, Args... dims, bool requires_grad): requires_grad(requires_grad) {
    initData(values, dims...);
}


// Overload << operator 
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(";
    if (tensor.getShape().size() == 1) {
        os << tensor.getData()[0]; // Scalar value
    }
    else {
        os << "shape=";
        for (int dim : tensor.getShape()) {
            os << dim << "x";
        }
        os.seekp(-1, std::ios_base::end); // Remove trailing "x"
    }
    os << ", requires_grad=" << (tensor.requires_grad ? "true" : "false") << ")";
    return os;
}

const std::vector<int>& Tensor::getShape() const {
    return dims_;
}

const std::vector<double>& Tensor::getData() const {
    return data_;
}

template <typename T, typename... Args>
void Tensor::initData(const std::initializer_list<T>& values, int dim, Args... dims) {
    dims_.push_back(dim);
    data_.insert(data_.end(), values.begin(), values.end());
    if constexpr (sizeof...(dims) > 0) {
        for (const auto& value : values) {
            initData(value, dims...);
        }
    }
}