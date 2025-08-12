/*
 * mul.cpp - element-wise multiplication operation implementation
 * 
 * this file implements tensor multiplication with broadcasting support:
 * - forward pass: handles shape mismatches through automatic broadcasting
 * - backward pass: applies product rule for gradient computation
 * - loss computation: essential for squared error terms in mse loss
 */

#include "mul.hpp"
#include <stdexcept>
#include "../graph.hpp"
#include "../tensor.hpp"

extern Graph global_graph;

// constructor implementation for mul operation
// stores weak references to input tensors to prevent circular dependencies
MulOp::MulOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);
    inputs.push_back(b);
}

// backward pass computes gradients w.r.t. both inputs using product rule
// ∂(a*b)/∂a = b and ∂(a*b)/∂b = a (from calculus)
void MulOp::backward(Tensor& grad_output) {
    auto a_const = inputs[0].lock();
    auto b_const = inputs[1].lock();
    if (!a_const || !b_const) return;

    auto a = std::const_pointer_cast<Tensor>(a_const);
    auto b = std::const_pointer_cast<Tensor>(b_const);

    if (a->requires_grad) {
        if (a->grad.empty()) a->grad.resize(a->data.size(), 0.0f);
        for (size_t i = 0; i < a->data.size(); ++i)
            a->grad[i] += grad_output.grad[i] * b->data[i]; // ∂(a*b)/∂a = b
    }

    if (b->requires_grad) {
        if (b->grad.empty()) b->grad.resize(b->data.size(), 0.0f);
        for (size_t i = 0; i < b->data.size(); ++i)
            b->grad[i] += grad_output.grad[i] * a->data[i]; // ∂(a*b)/∂b = a
    }

    // propagate gradients to input tensors' creators to maintain gradient chain
    // critical fix: check if creators are different to prevent infinite loops
    auto a_creator = a->creator.lock();
    if (a_creator && a_creator.get() != this) {
        a_creator->backward(*a);
    }
    auto b_creator = b->creator.lock();
    if (b_creator && b_creator.get() != this) {
        b_creator->backward(*b);
    }
}

std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // determine output shape and handle broadcasting for shape mismatches
    // this enables operations between tensors of different shapes
    std::vector<int> output_shape;
    bool needs_broadcasting = false;
    
    if (a->shape.size() != b->shape.size()) {
        // different dimensions - assume broadcasting is needed
        needs_broadcasting = true;
        if (a->shape.size() > b->shape.size()) {
            output_shape = a->shape;
        } else {
            output_shape = b->shape;
        }
    } else {
        // same dimensions - check if shapes match
        output_shape = a->shape;
        for (size_t i = 0; i < a->shape.size(); ++i) {
            if (a->shape[i] != b->shape[i]) {
                needs_broadcasting = true;
                output_shape[i] = std::max(a->shape[i], b->shape[i]);
            }
        }
    }
    
    auto result = std::make_shared<Tensor>(output_shape, a->requires_grad || b->requires_grad);
    
    if (needs_broadcasting) {
        // handle broadcasting: 2d * 1d (matrix * bias vector)
        // this pattern is less common than addition but still supported
        if (a->shape.size() == 2 && b->shape.size() == 1) {
            // matrix * bias vector: broadcast bias across rows
            int rows = a->shape[0];
            int cols = a->shape[1];
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result->data[i * cols + j] = a->data[i * cols + j] * b->data[j];
                }
            }
        } else if (a->shape.size() == 1 && b->shape.size() == 2) {
            // bias vector * matrix: broadcast bias across rows
            int rows = b->shape[0];
            int cols = b->shape[1];
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result->data[i * cols + j] = a->data[j] * b->data[i * cols + j];
                }
            }
        } else {
            // fallback for other broadcasting cases
            for (size_t i = 0; i < result->data.size(); ++i) {
                result->data[i] = a->data[i % a->data.size()] * b->data[i % b->data.size()];
            }
        }
    } else {
        // no broadcasting needed - direct multiplication
        for (size_t i = 0; i < result->data.size(); ++i)
            result->data[i] = a->data[i] * b->data[i];
    }

    if (result->requires_grad) {
        // create mul operation and integrate with computational graph
        auto op = std::make_shared<MulOp>(a, b);
        result->set_creator(op);

        // register with global graph for lifetime management
        global_graph.add_tensor(result);
        global_graph.add_op(op);
    }

    return result;
}
