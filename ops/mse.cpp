#include "mse.hpp"
#include <iostream>
#include "sub.hpp"
#include "mean.hpp"
#include "mul.hpp"
#include "../tensor_ops.hpp"
#include "../graph.hpp"

extern Graph global_graph;

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& prediction, const std::shared_ptr<Tensor>& target) {
    auto diff = prediction - target;                      // SubOp
    auto squared = (diff) * (diff);                       // MulOp
    auto loss = squared->mean();    // MeanOp

    std::cout << "[mse_loss] returning loss tensor: " << loss.get() << "\n";
    if (loss->creator.expired()) {
        std::cout << "[mse_loss] WARNING: creator expired\n";
    } else {
        std::cout << "[mse_loss] creator set\n";
    }

    // Register ops and tensors in global graph
    global_graph.add_tensor(diff);
    global_graph.add_tensor(squared);
    global_graph.add_tensor(loss);
    return loss;
}
