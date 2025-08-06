#pragma once

#include "../src/module.hpp"
#include <vector>
#include <memory>

class Sequential : public Module {
public:
    Sequential() = default;

    void add_module(std::shared_ptr<Module> module);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    std::vector<std::shared_ptr<Tensor>> parameters() const override;

    std::string name() const override { return "Sequential"; }

private:
    std::vector<std::shared_ptr<Module>> modules;
};
