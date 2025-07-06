#include "add.hpp"

AddOp::AddOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);
    inputs.push_back(b);
}

void AddOp::backward(Tensor& grad_output) {
    for (auto& input : inputs) {
        if (input->requires_grad) {
            if (input->grad.empty()) {
                input->grad.resize(input->data.size(), 0.0f);
            }
            for (size_t i = 0; i < grad_output.grad.size(); ++i) {
                input->grad[i] += grad_output.grad[i];
            }

            if (input->creator) {
                input->backward();
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
