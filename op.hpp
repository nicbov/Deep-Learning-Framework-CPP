/*
 * op.hpp - base operation class for computational graph
 * 
 * this class defines the interface for all operations in the neural network:
 * - stores weak references to input tensors to avoid circular dependencies
 * - provides virtual backward() method for gradient computation
 * - enables automatic differentiation through the computation graph
 * 
 * IMPORTANT design insight: operations are the nodes in the computational graph
 * each operation knows how to compute gradients w.r.t. its inputs
 */

#pragma once
#include <vector>
#include <memory>
class Tensor;

class Op : public std::enable_shared_from_this<Op> {
public:
    // weak references to input tensors to prevent circular ownership
    // weak_ptr allows tensors to be destroyed when no longer needed
    std::vector<std::weak_ptr<Tensor>> inputs;

    // compute gradients w.r.t. input tensors during backpropagation
    // grad_output contains gradients flowing backward from output
    virtual void backward(Tensor& grad_output) = 0;
    
    virtual ~Op() = default;
};
