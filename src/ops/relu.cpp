#include "relu.hpp"

ReLUOp::ReLUOp(std::shared_ptr<Tensor> input) {
    inputs.push_back(input);
}

void ReLUOp::backward(Tensor& grad_output) {
    auto input = inputs[0];
    if (!input->requires_grad) return;

    if (input->grad.empty()) {
        input->grad.resize(input->data.size(), 0.0f);
    }

    for (size_t i = 0; i < input->data.size(); ++i) {
        float grad = input->data[i] > 0.0f ? 1.0f : 0.0f;
        input->grad[i] += grad * grad_output.grad[i];
    }

    if (input->creator) {
        input->backward();
    }
}

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, input->requires_grad);
    for (size_t i = 0; i < input->data.size(); ++i) {
        result->data[i] = std::max(0.0f, input->data[i]);
    }

    if (result->requires_grad) {
        auto op = std::make_shared<ReLUOp>(input);
        result->creator = op;
    }

    return result;
}
