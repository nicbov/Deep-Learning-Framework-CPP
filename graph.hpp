#pragma once
#include <vector>
#include <memory>
#include "tensor.hpp"
#include "op.hpp"

// Manages strong ownership of Tensors and Ops in the computation graph.
class Graph {
public:
    // Store shared_ptrs to keep alive all Tensors and Ops created during forward
    std::vector<std::shared_ptr<Tensor>> tensors;
    std::vector<std::shared_ptr<Op>> ops;

    void add_tensor(const std::shared_ptr<Tensor>& t) {
        tensors.push_back(t);
    }

    void add_op(const std::shared_ptr<Op>& op) {
        ops.push_back(op);
    }

    // Clear all references to allow destruction and free memory
    void clear() {
        tensors.clear();
        ops.clear();
    }
};
