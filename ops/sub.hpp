/*
 * sub.hpp - element-wise subtraction operation for tensors
 * 
 * this header defines the subtraction operation used in loss computation:
 * - forward pass: element-wise subtraction between tensors
 * - backward pass: gradients flow directly to inputs (subtraction is linear)
 * - loss computation: critical for computing prediction - target differences
 */

#pragma once
#include "../tensor.hpp"
#include "../op.hpp"
#include "../graph.hpp"

extern Graph global_graph;

// sub operation handles element-wise subtraction between tensors
// primarily used in loss computation: prediction - target
class SubOp : public Op {
    std::shared_ptr<Tensor> a_keep_alive;
    std::shared_ptr<Tensor> b_keep_alive;

public:
    SubOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward(Tensor& grad_output) override;
};

// global sub function creates sub operations and integrates with computational graph
// this is the user-facing interface for tensor subtraction operations
std::shared_ptr<Tensor> sub(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
