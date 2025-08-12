/*
 * div.cpp - scalar division operation implementation
 * 
 * this file implements tensor division by scalar values:
 * - forward pass: element-wise division by scalar with safety checks
 * - backward pass: gradient follows chain rule: ∂(a/c)/∂a = 1/c
 * - normalization: essential for scaling tensors and loss computation
 */

#include "div.hpp"
#include "../op.hpp"
#include "../tensor.hpp"
#include "../graph.hpp"
#include <iostream>
#include <memory>

extern Graph global_graph;

// constructor stores input tensor reference and scalar divisor
// weak references prevent circular dependencies while maintaining access to input data
DivOp::DivOp(std::shared_ptr<Tensor> input, float scalar_) : scalar(scalar_) {
    inputs.push_back(std::const_pointer_cast<Tensor>(input));
}

void DivOp::backward(Tensor& grad_output) {
    auto input_const = inputs[0].lock();
    if (!input_const) return;

    auto input = std::const_pointer_cast<Tensor>(input_const);
    if (input->requires_grad) {
        if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
        for (size_t i = 0; i < input->data.size(); ++i)
            input->grad[i] += grad_output.grad[i] / scalar; // ∂(a/c)/∂a = 1/c
    }

    // propagate gradients to input tensor's creator to maintain gradient chain
    // critical fix: check if creator is different to prevent infinite loops
    auto input_creator = input->creator.lock();
    if (input_creator && input_creator.get() != this) {
        // only propagate if the creator is different from this operation
        input_creator->backward(*input);
    }
}

std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> input, float scalar) {
    // safety check: prevent division by zero which would cause undefined behavior
    if (scalar == 0.0f) throw std::runtime_error("div: division by zero");

    // create output tensor with same shape and gradient requirements as input
    auto result = std::make_shared<Tensor>(input->shape, input->requires_grad);
    result->data.resize(input->data.size());

    // perform element-wise division by scalar
    for (size_t i = 0; i < result->data.size(); ++i)
        result->data[i] = input->data[i] / scalar;

    if (result->requires_grad) {
        // create div operation and integrate with computational graph
        auto op = std::make_shared<DivOp>(input, scalar);
        result->set_creator(op);

        // register with global graph for lifetime management
        global_graph.add_tensor(result);
        global_graph.add_op(op);
    }

    return result;
}
