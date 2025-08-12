/*
 * sequential.hpp - sequential container for neural network layers
 */

#pragma once

#include "../src/module.hpp"
#include <vector>
#include <memory>

class Sequential : public Module {
public:
    Sequential() = default;
    // adds a module to the sequential container
    // modules will always be executed in the order they were added
    void add_module(std::shared_ptr<Module> module);

    // forward pass: applies all modules sequentially to input
    // each module's output becomes the next module's input(this is opposite on backwards passes)
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    // collects parameters from all contained modules
    // returns concatenated list of all trainable parameters
    std::vector<std::shared_ptr<Tensor>> parameters() const override;

    // model identification for debugging and inspection
    std::string name() const override { return "Sequential"; }

private:
    // ordered list of modules to execute sequentially(this is used for debugging and inspection)
    std::vector<std::shared_ptr<Module>> modules;
};
