#include "mul.hpp"

MulOp::MulOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);
    inputs.push_back(b);
}

void MulOp::backward(Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    for (size_t i = 0; i < grad_output.grad.size(); ++i) {
        if (a->requires_grad) {
            if (a->grad.empty()) a->grad.resize(a->data.size(), 0.0f);
            a->grad[i] += grad_output.grad[i] * b->data[i];
        }
        if (b->requires_grad) {
            if (b->grad.empty()) b->grad.resize(b->data.size(), 0.0f);
            b->grad[i] += grad_output.grad[i] * a->data[i];
        }
    }

    if (a->creator) a->backward();
    if (b->creator) b->backward();
}

std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto result = std::make_shared<Tensor>(a->shape, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < result->data.size(); ++i) {
        result->data[i] = a->data[i] * b->data[i];
    }

    if (result->requires_grad) {
        auto op = std::make_shared<MulOp>(a, b);
        result->creator = op;
    }

    return result;
}
