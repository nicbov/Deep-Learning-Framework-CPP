#include "tensor.hpp"
#include "op.hpp"

Tensor::Tensor(std::vector<int> shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_) {
    int total = numel();
    data.resize(total, 0.0f);
    if (requires_grad) {
        grad.resize(total, 0.0f);
    }
}

int Tensor::numel() const {
    int n = 1;
    for (int d : shape) n *= d;
    return n;
}

void Tensor::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
}

void Tensor::backward() {
    if (!requires_grad) {
        throw std::runtime_error("Cannot call backward on tensor without requires_grad.");
    }

    std::fill(grad.begin(), grad.end(), 1.0f);

    if (creator) {
        creator->backward(*this);
    }
}


void Tensor::print_data() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 != shape.size()) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i + 1 != data.size()) std::cout << ", ";
    }
    std::cout << "])\n";
}
