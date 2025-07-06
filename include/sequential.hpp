#pragma once

#include "module.hpp"
#include <vector>
#include <memory>

class Sequential : public Module {
public:
    Sequential() = default;

    // Add a module (layer) to the sequence
    void add_module(std::shared_ptr<Module> module);

    // Forward input through all modules
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    // Return all parameters from all modules
    std::vector<std::shared_ptr<Tensor>> parameters() const override;

    std::string name() const override { return "Sequential"; }

private:
    std::vector<std::shared_ptr<Module>> modules;
};
