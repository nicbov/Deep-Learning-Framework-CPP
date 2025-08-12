/*
 * add.hpp - element-wise addition operation for tensors
 * 
 * this header defines the addition operation which is fundamental to neural networks:
 * - forward pass: element-wise addition with automatic broadcasting support
 * - backward pass: gradients flow directly to inputs (addition is linear)
 * - broadcasting: handles matrix + bias vector operations efficiently
 * 
 * IMPORTANT: broadcasting support makes it crucial for bias addition in linear layers
 */

#pragma once
#include "../tensor.hpp"
#include "../op.hpp"

// add operation handles element-wise addition between tensors
// supports broadcasting for common neural network patterns like matrix + bias
class AddOp : public Op {
public:
    AddOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward(Tensor& grad_output) override;
};

// global add function creates add operations and integrates with computational graph
// this is the user-facing interface for tensor addition operations
std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
