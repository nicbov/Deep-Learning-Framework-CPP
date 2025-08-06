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

    for (size_t i = 0; i < input->data.size(); ++i) {
        input->grad[i] += exponent * std::pow(input->data[i], exponent - 1) * grad_output.grad[i];
    }

    if (auto creator = input->creator.lock()) creator->backward(*input);
}

