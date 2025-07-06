#include "mse.hpp"
#include <stdexcept>
#include <cmath>

MSELossOp::MSELossOp(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target)
    : target(target) {
    inputs.push_back(pred);
}

void MSELossOp::backward(Tensor& grad_output) {
    auto pred = inputs[0];
    size_t n = pred->data.size();

    if (pred->requires_grad) {
        if (pred->grad.empty()) pred->grad.resize(n, 0.0f);
        for (size_t i = 0; i < n; ++i) {
            pred->grad[i] += (2.0f / n) * (pred->data[i] - target->data[i]);
        }
    }

    if (pred->creator) {
        pred->backward();
    }
}

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target) {
    if (pred->shape != target->shape)
        throw std::runtime_error("Shape mismatch in mse_loss.");

    float loss_val = 0.0f;
    for (size_t i = 0; i < pred->data.size(); ++i) {
        float diff = pred->data[i] - target->data[i];
        loss_val += diff * diff;
    }
    loss_val /= pred->data.size();

    auto result = std::make_shared<Tensor>(std::vector<int>{1}, pred->requires_grad);
    result->data[0] = loss_val;

    if (result->requires_grad) {
        auto op = std::make_shared<MSELossOp>(pred, target);
        result->creator = op;
    }

    return result;
}
