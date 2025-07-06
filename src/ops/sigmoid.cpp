#include "sigmoid.hpp"
#include <cmath>

SigmoidOp::SigmoidOp(std::shared_ptr<Tensor> input) {
    inputs.push_back(input);
}

void SigmoidOp::backward(Tensor& grad_output) {
    auto input = inputs[0];
    if (input->requires_grad) {
        if (input->grad.empty()) {
            input->grad.resize(input->data.size(), 0.0f);
        }
        // Gradient of sigmoid: s * (1 - s)
        for (size_t i = 0; i < grad_output.grad.size(); ++i) {
            float s = grad_output.data[i];
            float grad_val = grad_output.grad[i] * s * (1.0f - s);
            input->grad[i] += grad_val;
        }
        if (input->creator) {
            input->backward();
        }
    }
}

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> input) {
    auto result = std::make_shared<Tensor>(input->shape, input->requires_grad);
    for (size_t i = 0; i < input->data.size(); ++i) {
        result->data[i] = 1.0f / (1.0f + std::exp(-input->data[i]));
    }
    if (result->requires_grad) {
        auto op = std::make_shared<SigmoidOp>(input);
        result->creator = op;
    }
    return result;
}
