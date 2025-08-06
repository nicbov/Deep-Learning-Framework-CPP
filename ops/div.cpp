#include "div.hpp"
#include "../op.hpp"
#include "../tensor.hpp"
#include "../graph.hpp"
#include <iostream>
#include <memory>

extern Graph global_graph;

DivOp::DivOp(std::shared_ptr<Tensor> input, float scalar_) : scalar(scalar_) {
    inputs.push_back(std::const_pointer_cast<Tensor>(input));
}

void DivOp::backward(Tensor& grad_output) {
    auto input_const = inputs[0].lock();
    if (!input_const) return;

    auto input = std::const_pointer_cast<Tensor>(input_const);
    if (!input->requires_grad) return;

    if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);

    for (size_t i = 0; i < input->grad.size(); ++i)
        input->grad[i] += grad_output.grad[i] / scalar;

    if (auto c = input->creator.lock()) c->backward(*input);
}

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> input, float scalar) {
    if (scalar == 0.0f) throw std::runtime_error("div: division by zero");

    auto result = std::make_shared<Tensor>(input->shape, input->requires_grad);
    result->data.resize(input->data.size());

    for (size_t i = 0; i < result->data.size(); ++i)
        result->data[i] = input->data[i] / scalar;

    if (result->requires_grad) {
        auto op = std::make_shared<DivOp>(input, scalar);
        result->creator = op;

        global_graph.add_tensor(result);
        global_graph.add_op(op);
    }

    return result;
}
