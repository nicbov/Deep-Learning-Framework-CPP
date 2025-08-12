/*
 * tensor.hpp - core tensor class for automatic differentiation
 * 
 * this class implements the fundamental data structure for neural network computations:
 * - multi-dimensional arrays with automatic shape management
 * - gradient tracking for backpropagation (requires_grad flag)
 * - computational graph linkage through creator pointers
 * - overloaded operators for building computation graphs
 * 
 * design insight: tensors are immutable - operations create new tensors
 * gradient buffers are allocated on-demand to save memory
 */

#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <typeinfo>

class Op;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    // tensor shape and data storage
    std::vector<int> shape;           // dimensions (e.g., [batch_size, features])
    std::vector<float> data;          // actual numerical values stored contiguously

    // automatic differentiation support
    bool requires_grad = false;        // whether this tensor participates in gradient computation
    std::vector<float> grad;          // gradients w.r.t. this tensor (allocated on-demand)

    // computational graph linkage
    std::weak_ptr<Op> creator;        // operation that created this tensor (for backprop)

    // tensor operations - all create new tensors and maintain gradient chains
    std::shared_ptr<Tensor> operator-(const Tensor& other) const;  // element-wise subtraction
    std::shared_ptr<Tensor> operator*(const Tensor& other) const;  // element-wise multiplication
    std::shared_ptr<Tensor> operator+(const Tensor& other) const;  // element-wise addition
    std::shared_ptr<Tensor> pow(float exponent) const;             // element-wise power
    std::shared_ptr<Tensor> operator/(float scalar) const;         // scalar division
    std::shared_ptr<Tensor> matmul(const Tensor& other) const;     // matrix multiplication
    std::shared_ptr<Tensor> mean() const;                          // reduction to scalar

    // construction and memory management
    Tensor(std::vector<int> shape, bool requires_grad = false);

    // gradient computation support
    int numel() const;                // total number of elements (product of shape)
    void zero_grad();                 // reset gradients to zero (called before each forward pass)
    void backward();                  // initiate backpropagation from this tensor
    void print_data() const;          // debug output of tensor contents
    std::shared_ptr<Tensor> detach() const;  // create tensor copy without gradient tracking

    // computational graph management
    void set_creator(std::shared_ptr<Op> op);  // link to operation that created this tensor
};
