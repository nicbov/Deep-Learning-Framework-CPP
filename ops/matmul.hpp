/*
 * matmul.hpp - matrix multiplication operation for neural network computations
 * 
 * this operation is fundamental to neural networks as it implements the core linear transformation:
 * - computes c = a * b where a is [m, k] and b is [k, n], resulting in [m, n]
 * - critical for forward pass: input features are transformed by weight matrices
 * - essential for backpropagation: gradients flow through matrix multiplication chains
 * - used extensively in linear layers, attention mechanisms, and feature transformations
 * 
 * IMPORTANT insight: matrix multiplication is the primary computational bottleneck in neural networks
 * this operation dominates training time and memory usage for large models
 */

#pragma once
#include "../tensor.hpp"
#include "../op.hpp"

// matrix multiplication operation that maintains computational graph for automatic differentiation
// stores weak references to input tensors to prevent circular dependencies while enabling gradient flow
class MatMulOp : public Op{
public:
    // constructor stores input tensors for gradient computation during backpropagation
    // inputs are stored as weak_ptr to avoid circular ownership issues
    MatMulOp(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
    
    // computes gradients w.r.t. input tensors using the chain rule
    // grad_output contains gradients flowing backward from the output tensor
    void backward(Tensor& grad_output) override;
};

// convenience function that creates matrix multiplication operation and registers with computation graph
// this is the primary interface used by neural network layers for linear transformations
std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
