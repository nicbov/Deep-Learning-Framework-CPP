#include "tanh.hpp"
#include <cmath>

TanhOp::TanhOp(std::shared_ptr<Tensor> input) {
    inputs.push_back(input);
}

void TanhOp::backward(Tensor& grad_output) {
    auto input = inputs[0];
    if (input->requires_grad) {
        if (input->grad.empty()) {
            input->grad.resize(input->data.size(), 0.0f);
        }
        // Gradient of tanh: 1 - tanh^2(x)
        for (size_t i = 0; i < grad_output.grad.size(); ++i) {
            float t = grad_output.data[i];
            float grad_val = grad_output.grad[i] * (1.0f - t * t);
            input->grad[i] += grad_val;
        }
        if (input->creator) {
            input->backward();
        }
    }
}

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, input->requires_grad);
    for (size_t i = 0; i < input->data.size(); ++i) {
        result->data[i] = std::tanh(input->data[i]);
    }
    if (result->requires_grad) {
        auto op = std::make_shared<TanhOp>(input);
        result->creator = op;
    }
    return result;
}
