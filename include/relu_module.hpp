#pragma once

#include "module.hpp"
#include "tensor.hpp"

class ReLU : public Module {
public:
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    std::vector<std::shared_ptr<Tensor>> parameters() const override {
        return {};  // ReLU has no parameters
    }

    std::string name() const override { return "ReLU"; }
};
