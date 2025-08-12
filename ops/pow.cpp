#include "../tensor.hpp"
#include "pow.hpp"
#include <cmath>
#include "../graph.hpp"

extern Graph global_graph;

PowOp::PowOp(const std::shared_ptr<Tensor>& input, float exponent_) : exponent(exponent_) {
    auto input_nc = std::const_pointer_cast<Tensor>(input);
    inputs.push_back(input_nc);

    // Keep input alive
    input_keep_alive = input_nc;
}

void PowOp::backward(Tensor& grad_output) {
    auto input_const = inputs[0].lock();
    if (!input_const) return;

    auto input = std::const_pointer_cast<Tensor>(input_const);
    if (!input->requires_grad) return;

    if (input->grad.size() != input->data.size())
        input->grad.assign(input->data.size(), 0.0f);

    if (input->requires_grad) {
        if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
        for (size_t i = 0; i < input->data.size(); ++i) {
            input->grad[i] += exponent * std::pow(input->data[i], exponent - 1) * grad_output.grad[i]; // Changed from = to += for gradient accumulation
        }
    }

    // CRITICAL FIX: Propagate gradients to input tensor's creator
    // But avoid infinite loops by checking if the creator is different
    auto input_creator = input->creator.lock();
    if (input_creator && input_creator.get() != this) {
        // Only propagate if the creator is different from this operation
        input_creator->backward(*input);
    }
}

