/*
 * linear.hpp - fully connected linear layer interface
 * 
 * this header defines the linear transformation layer which is fundamental to neural networks:
 * - parameter management: weights and biases are trainable parameters
 * - forward computation: y = xW + b with matrix multiplication and bias addition
 * - computational graph integration: creates operations for automatic differentiation
 * 
 * IMPORTANT design insight: weights and biases are initialized as trainable parameters
 * the forward pass creates matmul and add operations to maintain gradient flow
 */

#pragma once

#include "src/module.hpp"
#include <memory>
#include <vector>

// linear layer implements the standard fully connected transformation
// this is the most common layer type in neural networks for feature transformation
class Linear : public Module {
public:
    // trainable parameters - initialized with he initialization for relu activations
    // these tensors get updated by the optimizer during training steps
    std::shared_ptr<Tensor> weight;  // input_features × output_features matrix
    std::shared_ptr<Tensor> bias;    // output_features vector

    // constructor initializes weights and biases with proper initialization
    // he initialization prevents vanishing gradients in deep networks
    Linear(int in_features, int out_features);

    // forward pass: computes y = xW + b
    // creates computational graph operations for automatic differentiation
    // input shape: [batch_size, in_features], output shape: [batch_size, out_features]
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    // returns all trainable parameters for optimizer access
    // optimizer needs these to update weights and biases during training
    std::vector<std::shared_ptr<Tensor>> parameters() const override;

    // layer identification for debugging and model inspection
    // useful for understanding network architecture and parameter counts
    std::string name() const override { return "Linear"; }

    // store created operations to maintain computational graph
    // note: this is currently unused but provides extensibility for future features
    std::vector<std::shared_ptr<Op>> ops;
};
