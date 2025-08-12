/*
 * relu.hpp - rectified linear unit activation function interface
 * 
 * this header defines the relu activation function which is essential for neural networks:
 * - module interface: inherits from base module class for consistent layer behavior
 * - operation separation: distinguishes between module interface and computation implementation
 * - parameter-free: relu has no trainable parameters, only activation computation
 * 
 * design insight: separates the user-facing module interface from the internal operation
 * this allows for clean abstraction while maintaining computational graph integrity
 */

#pragma once

#include "src/module.hpp"
#include "op.hpp"
#include <memory>

// relu module provides the user interface for adding relu activation to neural networks
// this is what gets added to sequential models: model->add_module(std::make_shared<ReLU>())
class ReLU : public Module {
public:
    ReLU() = default;

    // forward pass: applies relu activation to input tensor
    // creates relu operation for proper gradient computation and graph integration
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    // relu has no trainable parameters (pure activation function)
    // this is important for optimizer behavior - no weight updates needed
    std::vector<std::shared_ptr<Tensor>> parameters() const override {
        return {}; // relu has no parameters
    }

    // layer identification for debugging and model inspection
    // useful for understanding network architecture during training
    std::string name() const override { return "ReLU"; }
};

// relu operation handles the actual computation and gradient computation
// this is the internal implementation that gets created during forward pass
class ReLUOp : public Op {
public:
    ReLUOp(std::shared_ptr<Tensor> input);
    
    // forward pass: computes relu(input) = max(0, input)
    // creates output tensor and stores it for backward pass
    std::shared_ptr<Tensor> forward();
    
    // backward pass: computes gradients w.r.t. input
    // gradient is 1 if input > 0, 0 otherwise (simple but effective)
    void backward(Tensor& grad_output) override;
    
    // operation identification for debugging and graph inspection
    // helps track computation flow during backpropagation
    std::string name() const { return "ReLU"; }
    
private:
    std::shared_ptr<Tensor> input;   // input tensor for gradient computation
    std::shared_ptr<Tensor> output;  // cached output for backward pass
};
