/*
 * div.hpp - scalar division operation for tensors
 * 
 * this header defines the division operation used for normalization and scaling:
 * - forward pass: element-wise division by scalar value
 * - backward pass: gradient follows chain rule: ∂(a/c)/∂a = 1/c
 * - normalization: commonly used for scaling tensors and loss computation
 */

#pragma once
#include "../tensor.hpp"
#include "../op.hpp"

// div operation handles element-wise division by scalar values
// primarily used for normalization and scaling operations
class DivOp : public Op {
public:
    DivOp(std::shared_ptr<Tensor> a, float scalar);
    void backward(Tensor& grad_output) override;
};

// global div function creates div operations and integrates with computational graph
// this is the user-facing interface for tensor division operations
std::shared_ptr<Tensor> div(std::shared_ptr<Tensor> a, float scalar);
