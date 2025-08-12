/*
 * module.hpp - base class for all neural network modules
 * 
 * this class defines the interface for neural network components:
 * - forward pass: input -> output computation
 * - parameter management: trainable weights and biases
 * - gradient zeroing: preparation for backpropagation
 * - module identification: names for debugging and inspection
 * 
 * IMPORTANT: all neural network layers inherit from this base class
 * this enables polymorphic behavior and consistent interface across different layer types
 */

#pragma once

#include <vector>
#include <memory>
#include <string>

#include "../tensor.hpp"

class Module {
public:
    virtual ~Module() = default;

    // forward computation: transforms input tensor to output tensor
    // this is the main computation performed during inference and training
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;

    // collect trainable parameters (weights, biases) for optimizer access
    // parameters are tensors with requires_grad=true that get updated during training
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const = 0;

    // zero gradients of all parameters before each forward pass
    // this prevents gradient accumulation across multiple backward passes
    virtual void zero_grad();

    // module name for debugging, logging, and model inspection
    // useful for identifying layers in complex architectures
    virtual std::string name() const { return "Module"; }
};
