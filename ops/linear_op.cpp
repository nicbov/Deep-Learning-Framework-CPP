// Author: Nico Boving
// linear operation
#include "linear_op.hpp"
#include <iostream>
#include <algorithm>  
#include "../graph.hpp"
#include <memory>

extern Graph global_graph;

LinearOp::LinearOp(const std::shared_ptr<Tensor>& input_,
                   const std::shared_ptr<Tensor>& weight_,
                   const std::shared_ptr<Tensor>& bias_)
    : input(input_), weight(weight_), bias(bias_) {
    inputs.push_back(input);
    inputs.push_back(weight);
    if (bias) inputs.push_back(bias);
}

void LinearOp::backward(Tensor& grad_output) {
    int batch = input->shape[0];
    int in_dim = input->shape[1];
    int out_dim = weight->shape[1];

    // Check grad_output size matches batch*out_dim
    if (grad_output.grad.size() != batch * out_dim) {
        std::cerr << "[LinearOp] ERROR: grad_output.grad size mismatch\n";
        return;
    }

    // Cast away const to mutate grad buffers
    auto input_mut = std::const_pointer_cast<Tensor>(input);
    auto weight_mut = std::const_pointer_cast<Tensor>(weight);
    std::shared_ptr<Tensor> bias_mut = bias ? std::const_pointer_cast<Tensor>(bias) : nullptr;

    if (input_mut->requires_grad) {
        if (input_mut->grad.size() != input_mut->data.size())
            input_mut->grad.resize(input_mut->data.size(), 0.0f);

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < in_dim; ++i) {
                float grad_val = 0.0f;
                for (int j = 0; j < out_dim; ++j) {
                    grad_val += grad_output.grad[b * out_dim + j] * weight_mut->data[i * out_dim + j];
                }
                input_mut->grad[b * in_dim + i] += grad_val;
            }
        }
    }

    if (weight_mut->requires_grad) {
        if (weight_mut->grad.size() != weight_mut->data.size())
            weight_mut->grad.resize(weight_mut->data.size(), 0.0f);

        for (int i = 0; i < in_dim; ++i) {
            for (int j = 0; j < out_dim; ++j) {
                float grad_val = 0.0f;
                for (int b = 0; b < batch; ++b) {
                    grad_val += input->data[b * in_dim + i] * grad_output.grad[b * out_dim + j];
                }
                weight_mut->grad[i * out_dim + j] += grad_val;
            }
        }
    }

    if (bias_mut && bias_mut->requires_grad) {
        if (bias_mut->grad.size() != bias_mut->data.size())
            bias_mut->grad.resize(bias_mut->data.size(), 0.0f);

        for (int j = 0; j < out_dim; ++j) {
            float grad_val = 0.0f;
            for (int b = 0; b < batch; ++b) {
                grad_val += grad_output.grad[b * out_dim + j];
            }
            bias_mut->grad[j] += grad_val;
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
