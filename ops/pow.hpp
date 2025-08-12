/*
 * pow.hpp - element-wise power operation for tensor computations
 * 
 * this operation implements element-wise exponentiation which is crucial for neural networks:
 * - computes y = x^exponent for each element in the input tensor
 * - essential for loss functions (e.g., mse uses power of 2), activation functions, and normalization
 * - enables polynomial transformations and non-linear feature engineering
 * - critical for implementing various mathematical operations in deep learning frameworks
 * 
 * Notice: power operations are often used with small exponents (1, 2, 0.5) for numerical stability
 * large exponents can lead to overflow/underflow issues during training
 */

#pragma once
#include "../op.hpp"
#include <memory>

// element-wise power operation that maintains computational graph for automatic differentiation
// stores the exponent value and input tensor reference for gradient computation
class PowOp : public Op {
public:
    // the exponent to raise each input element to (typically 2 for mse loss)
    float exponent;
    
    // constructor stores input tensor and exponent for backpropagation
    // input_keep_alive prevents premature destruction during computation
    PowOp(const std::shared_ptr<Tensor>& input, float exponent_);
    
    // strong reference to input tensor to ensure it remains alive during computation
    // this prevents issues with weak_ptr expiration in complex computational graphs
    std::shared_ptr<Tensor> input_keep_alive;

    // computes gradients w.r.t. input tensor using the power rule: d/dx(x^n) = n*x^(n-1)
    // grad_output contains gradients flowing backward from the output tensor
    void backward(Tensor& grad_output) override;
};
