#pragma once
#include <memory>
#include "tensor.hpp"  

// Binary operators on shared_ptr<Tensor>

inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return (*a) + (*b);
}

inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return (*a) - (*b);
}

inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return (*a) * (*b);
}

inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float scalar) {
    return (*a) / scalar;
}

inline std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, float exponent) {
    return a->pow(exponent);
}

inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a) {
    return a->mean();
}
