/*
 * add.cpp - element-wise addition operation implementation
 * 
 * this file implements tensor addition with sophisticated broadcasting support:
 * - forward pass: handles shape mismatches through automatic broadcasting
 * - backward pass: propagates gradients with proper broadcasting rules
 */

#include "add.hpp"
#include <stdexcept>
#include "../graph.hpp"
#include "../tensor.hpp"
#include <iostream>

extern Graph global_graph;

// constructor stores weak references to input tensors to prevent circular dependencies
// weak_ptr allows tensors to be destroyed when no longer needed
AddOp::AddOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);
    inputs.push_back(b);
}

void AddOp::backward(Tensor& grad_output) {
    for (auto& weak_input : inputs) {
        auto input = std::const_pointer_cast<Tensor>(weak_input.lock());
        if (!input) throw std::runtime_error("AddOp: input expired");

        if (input->requires_grad) {
            if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
            
            // handle broadcasting in backward pass for bias addition
            // bias tensors (1d) need gradients summed across batch dimension
            if (input->shape.size() == 1 && grad_output.shape.size() == 2) {
                // bias tensor (1d) - sum gradients across batch dimension
                for (size_t i = 0; i < input->data.size(); ++i) {
                    for (size_t batch = 0; batch < grad_output.shape[0]; ++batch) {
                        input->grad[i] += grad_output.grad[batch * grad_output.shape[1] + i];
                    }
                }
            } else {
                // same shape tensors - direct gradient assignment
                for (size_t i = 0; i < input->data.size(); ++i)
                    input->grad[i] += grad_output.grad[i];
            }

            // propagate gradients to input tensor's creator to maintain gradient chain
            // critical fix: check if creator is different to prevent infinite loops
            auto input_creator = input->creator.lock();
            if (input_creator && input_creator.get() != this) {
                // only propagate if the creator is different from this operation
                input_creator->backward(*input);
            }
        }
    }
}

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    // determine output shape and handle broadcasting for shape mismatches
    // this enables efficient operations like matrix + bias vector
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
        // handle broadcasting: 2d + 1d (matrix + bias vector)
        // this is the most common case in neural networks
        if (a->shape.size() == 2 && b->shape.size() == 1) {
            // matrix + bias vector: broadcast bias across rows
            int rows = a->shape[0];
            int cols = a->shape[1];
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result->data[i * cols + j] = a->data[i * cols + j] + b->data[j];
                }
            }
        } else if (a->shape.size() == 1 && b->shape.size() == 2) {
            // bias vector + matrix: broadcast bias across rows
            int rows = b->shape[0];
            int cols = b->shape[1];
            
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    result->data[i * cols + j] = a->data[j] + b->data[i * cols + j];
                }
            }
        } else {
            // fallback for other broadcasting cases
            for (size_t i = 0; i < result->data.size(); ++i) {
                result->data[i] = a->data[i % a->data.size()] + b->data[i % b->data.size()];
            }
        }
    } else {
        // no broadcasting needed - direct addition
        for (size_t i = 0; i < result->data.size(); ++i)
            result->data[i] = a->data[i] + b->data[i];
    }

    if (result->requires_grad) {
        // create add operation and integrate with computational graph
        auto op = std::make_shared<AddOp>(a, b);
        result->set_creator(op);

        // register with global graph for lifetime management
        global_graph.add_tensor(result);
        global_graph.add_op(op);
    }

    return result;
}
