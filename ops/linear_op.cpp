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
    std::cout << "[LinearOp] backward called\n";

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
            input_mut->grad.assign(input_mut->data.size(), 0.0f);
        else
            std::fill(input_mut->grad.begin(), input_mut->grad.end(), 0.0f);

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < in_dim; ++i) {
                float grad_val = 0.0f;
                for (int j = 0; j < out_dim; ++j) {
                    grad_val += grad_output.grad[b * out_dim + j] * weight_mut->data[i * out_dim + j];
                }
                input_mut->grad[b * in_dim + i] = grad_val;  // overwrite after zeroing
            }
        }

        std::cout << "[LinearOp] input grad sample: ";
        for (int i = 0; i < std::min(5, (int)input_mut->grad.size()); ++i) {
            std::cout << input_mut->grad[i] << " ";
        }
        std::cout << std::endl;
    }

    if (weight_mut->requires_grad) {
        if (weight_mut->grad.size() != weight_mut->data.size())
            weight_mut->grad.assign(weight_mut->data.size(), 0.0f);
        else
            std::fill(weight_mut->grad.begin(), weight_mut->grad.end(), 0.0f);

        for (int i = 0; i < in_dim; ++i) {
            for (int j = 0; j < out_dim; ++j) {
                float grad_val = 0.0f;
                for (int b = 0; b < batch; ++b) {
                    grad_val += input->data[b * in_dim + i] * grad_output.grad[b * out_dim + j];
                }
                weight_mut->grad[i * out_dim + j] = grad_val;  // overwrite after zeroing
            }
        }

        std::cout << "[LinearOp] weight grad sample: ";
        for (int i = 0; i < std::min(5, (int)weight_mut->grad.size()); ++i) {
            std::cout << weight_mut->grad[i] << " ";
        }
        std::cout << std::endl;
    }

    if (bias_mut && bias_mut->requires_grad) {
        if (bias_mut->grad.size() != bias_mut->data.size())
            bias_mut->grad.assign(bias_mut->data.size(), 0.0f);
        else
            std::fill(bias_mut->grad.begin(), bias_mut->grad.end(), 0.0f);

        for (int j = 0; j < out_dim; ++j) {
            float grad_val = 0.0f;
            for (int b = 0; b < batch; ++b) {
                grad_val += grad_output.grad[b * out_dim + j];
            }
            bias_mut->grad[j] = grad_val;  // overwrite after zeroing
        }

        std::cout << "[LinearOp] bias grad sample: ";
        for (int i = 0; i < std::min(5, (int)bias_mut->grad.size()); ++i) {
            std::cout << bias_mut->grad[i] << " ";
        }
        std::cout << std::endl;
    }

    if (auto creator = input->creator.lock()) {
        creator->backward(*input);
    }
}
