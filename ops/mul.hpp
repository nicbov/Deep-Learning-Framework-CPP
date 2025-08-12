/*
 * mul.hpp - element-wise multiplication operation for tensors
 * 
 * this header defines the multiplication operation used throughout neural networks:
 * - forward pass: element-wise multiplication between tensors
 * - backward pass: gradients use product rule: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
 * - loss computation: essential for squared error terms in mse loss
 */

#pragma once
#include "../tensor.hpp"
#include "../op.hpp"
#include "../tensor_ops.hpp"
#include "../graph.hpp"

extern Graph global_graph;

// mul operation handles element-wise multiplication between tensors
// used for hadamard product and squared error computation in loss functions
class MulOp : public Op {
    std::shared_ptr<Tensor> a_keep_alive;
    std::shared_ptr<Tensor> b_keep_alive;

public:
    MulOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward(Tensor& grad_output) override;
};

// global mul function creates mul operations and integrates with computational graph
// this is the user-facing interface for tensor multiplication operations
std::shared_ptr<Tensor> mul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);

