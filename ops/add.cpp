#include "add.hpp"
#include <stdexcept>

AddOp::AddOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);  // stored as weak_ptr in Op
    inputs.push_back(b);
}

void AddOp::backward(Tensor& grad_output) {
    for (auto& weak_input : inputs) {
        auto input = weak_input.lock();
        if (!input) {
            throw std::runtime_error("AddOp: input tensor expired");
        }

        if (input->requires_grad) {
            if (input->grad.empty()) {
                input->grad.resize(input->data.size(), 0.0f);
            }

            for (size_t i = 0; i < grad_output.grad.size(); ++i) {
                input->grad[i] += grad_output.grad[i];
            }

            auto creator_ptr = input->creator.lock();
        if (creator_ptr) {
    creator_ptr->backward(*input);
            }

        }
    }
}

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto result = std::make_shared<Tensor>(a->shape, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < result->data.size(); ++i) {
        result->data[i] = a->data[i] + b->data[i];
    }

    if (result->requires_grad) {
        auto op = std::make_shared<AddOp>(a, b);
        result->creator = op;
    }

    return result;
}
