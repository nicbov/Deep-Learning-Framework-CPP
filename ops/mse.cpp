#include "mse.hpp"
#include <iostream>

// Constructor stores weak_ptrs to inputs
MSELossOp::MSELossOp(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
    inputs.push_back(prediction);
    inputs.push_back(target);
}

// Backward pass: compute grad for prediction, then call backward on creator if any
void MSELossOp::backward(Tensor& grad_output) {
    auto pred_wp = inputs[0];
    auto target_wp = inputs[1];

    auto pred = pred_wp.lock();
    auto target = target_wp.lock();

    if (!pred || !target) {
        std::cerr << "[MSELossOp] One or more inputs expired. Skipping backward.\n";
        return;
    }

    if (grad_output.grad.empty()) {
        std::cerr << "[MSELossOp] grad_output.grad is empty! Initializing to 1.0\n";
        grad_output.grad = {1.0f}; // Failsafe
    }

    int n = pred->data.size();
    if (pred->requires_grad) {
        if (pred->grad.empty()) pred->grad.resize(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            pred->grad[i] += 2.0f * (pred->data[i] - target->data[i]) / n * grad_output.grad[0];
        }
    }

    // Call backward on creator if exists
    auto creator = pred->creator.lock();
    if (creator) {
        creator->backward(*pred);
    }
}

// mse_loss function that creates the loss tensor and sets creator
std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> prediction, std::shared_ptr<Tensor> target) {
    int n = prediction->data.size();

    // 🔍 Debug: check if tensors require grad
    std::cout << "[mse_loss] requires_grad: pred=" << prediction->requires_grad
              << ", target=" << target->requires_grad << std::endl;

    auto result = std::make_shared<Tensor>(
        std::vector<int>{1}, prediction->requires_grad || target->requires_grad);

    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = prediction->data[i] - target->data[i];
        sum += diff * diff;
    }
    result->data[0] = sum / n;

    if (result->requires_grad) {
        auto op = std::make_shared<MSELossOp>(prediction, target);
        result->creator = op;
    }    

    return result;
}
