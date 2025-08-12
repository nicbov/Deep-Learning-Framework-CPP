/*
 * tensor_ops.hpp - convenience operators for shared_ptr<tensor> operations
 * 
 * this header provides operator overloads that make tensor operations more intuitive:
 * - enables natural syntax like (a + b) instead of (*a + *b) for shared_ptr<tensor>
 * - maintains the same computational graph semantics as direct tensor operations
 * - simplifies neural network layer implementations and model construction
 * - all operations automatically register with the global computation graph for memory management
 * 
 * IMPORTANT insightt: these operators don't change the underlying computation
 * they make the code more readable while preserving automatic differentiation capabilities
 */

#pragma once
#include <memory>
#include "tensor.hpp"  

// convenience operators that work directly on shared_ptr<tensor> objects
// these operators dereference the pointers and call the corresponding tensor methods
// the resulting tensors are automatically registered with the global computation graph

// element-wise addition: creates add operation and maintains gradient flow
inline std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return (*a) + (*b);
}

// element-wise subtraction: creates sub operation and maintains gradient flow
inline std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return (*a) - (*b);
}

// element-wise multiplication: creates mul operation and maintains gradient flow
inline std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    return (*a) * (*b);
}

// scalar division: divides each tensor element by the scalar value
inline std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float scalar) {
    return (*a) / scalar;
}

// element-wise power: raises each tensor element to the specified exponent
inline std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, float exponent) {
    return a->pow(exponent);
}

// reduction to scalar: computes the mean of all tensor elements
inline std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a) {
    return a->mean();
}

