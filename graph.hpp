/*
 * graph.hpp - computational graph memory manager
 * 
 * this class manages the lifecycle of all tensors and operations during computation
 *
 * IMPORTANT: tensors and operations are created during forward pass
 * but must remain alive until after backward pass and optimizer step
 * this prevents the "dangling pointer" problem in automatic differentiation
 */

#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "op.hpp"

// manages strong ownership of tensors and operations in the computation graph
class Graph {
public:
    // store shared_ptrs to keep alive all tensors and operations created during forward pass
    // these references prevent premature destruction of intermediate computation results
    std::vector<std::shared_ptr<Tensor>> tensors;
    std::vector<std::shared_ptr<Op>> ops;

    // add tensor to graph for lifetime management
    void add_tensor(const std::shared_ptr<Tensor>& t) {
        tensors.push_back(t);
    }

    // add operation to graph for lifetime management
    void add_op(const std::shared_ptr<Op>& op) {
        ops.push_back(op);
    }

    // clear all references to allow destruction and free memory
    // called after optimizer step to prevent memory accumulation across epochs
    void clear() {
        tensors.clear();
        ops.clear();
    }
};
