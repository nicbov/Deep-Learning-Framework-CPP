#include "tensor.hpp"
#include "op.hpp"
#include <iostream>

Tensor::Tensor(std::vector<int> shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_) {
    int total = numel();
    data.resize(total, 0.0f);
    if (requires_grad) {
        grad.resize(total, 0.0f);
        std::cout << "[Tensor ctor] requires_grad=true, grad buffer size: " << grad.size() << std::endl;
    } else {
        std::cout << "[Tensor ctor] requires_grad=false" << std::endl;
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
        std::cout << "[Tensor] zero_grad called, grad zeroed" << std::endl;
    }
}

void Tensor::backward() {
    std::cout << "[Tensor] backward called on Tensor with shape: ";
    for (auto d : shape) std::cout << d << " ";
    std::cout << std::endl;

    if (!requires_grad) {
        std::cerr << "[Tensor] ERROR: backward() called on a tensor that does not require grad (possibly detached!)" << std::endl;
        throw std::runtime_error("Cannot call backward on tensor without requires_grad.");
    }

    if (grad.empty()) {
        grad.resize(data.size(), 1.0f); // 👈 MAKE SURE grad[0] is accessible!
        std::cout << "[Tensor] grad initialized to 1.0 for loss tensor" << std::endl;
    }

    if (auto creator_shared = creator.lock()) {
        std::cout << "[Tensor] calling creator->backward()" << std::endl;
        creator_shared->backward(*this);  // 👈 this receives grad[0] = 1.0f now
    } else {
        std::cout << "[Tensor] no creator found or expired, backward ends here" << std::endl;
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

std::shared_ptr<Tensor> Tensor::detach() const {
    std::cout << "[Tensor] detach() called, requires_grad set to false, creator removed" << std::endl;
    auto new_tensor = std::make_shared<Tensor>(shape, false); 
    new_tensor->data = data;
    return new_tensor;
}
