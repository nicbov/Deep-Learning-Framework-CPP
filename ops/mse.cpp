#include "mse.hpp"
#include <iostream>
#include "sub.hpp"
#include "mean.hpp"
#include "mul.hpp"
#include "../tensor_ops.hpp"
#include "../graph.hpp"

extern Graph global_graph;

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& prediction, const std::shared_ptr<Tensor>& target) {
    // Create the computation graph properly
    auto diff = prediction - target;                      // SubOp
    auto squared = (diff) * (diff);                       // MulOp
    auto loss = squared->mean();                          // MeanOp

    // The tensor operators should have already set up the creators and registered with global_graph
    // But let's ensure the loss tensor is properly set up
    if (prediction->requires_grad || target->requires_grad) {
        loss->requires_grad = true;
    }
    
    return loss;
}
