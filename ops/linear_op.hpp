/*
 * linear_op.hpp - linear transformation operation for neural network layers
 * 
 * this operation implements the fundamental linear transformation y = xW + b:
 * - combines matrix multiplication (xW) with bias addition (+b) in a single operation
 * - essential for implementing fully connected layers in neural networks
 * - stores input, weight, and bias tensors for gradient computation during backpropagation
 * - enables efficient computation of linear transformations with automatic differentiation
 */

#pragma once

#include <memory>
#include "../tensor.hpp"
#include "../op.hpp"

class Tensor;

// linear transformation operation that combines matrix multiplication and bias addition
// this operation is the core building block for fully connected neural network layers
class LinearOp : public Op {
public:
    // input tensor: typically [batch_size, input_features] from previous layer or input data
    std::shared_ptr<Tensor> input;  
    
    // weight matrix: [input_features, output_features] - the learnable transformation matrix
    std::shared_ptr<Tensor> weight;
    
    // bias vector: [output_features] - the learnable offset added to each output
    std::shared_ptr<Tensor> bias;

    // constructor stores all three tensors for gradient computation during backpropagation
    // these tensors are used to compute gradients w.r.t. weights, biases, and inputs
    LinearOp(const std::shared_ptr<Tensor>& input_,
             const std::shared_ptr<Tensor>& weight_,
             const std::shared_ptr<Tensor>& bias_);

    // computes gradients w.r.t. input, weight, and bias tensors using the chain rule
    // grad_output contains gradients flowing backward from the output tensor
    void backward(Tensor& grad_output) override;
};
