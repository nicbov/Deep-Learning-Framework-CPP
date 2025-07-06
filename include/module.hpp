#pragma once

#include <vector>
#include <memory>
#include <string>

#include "tensor.hpp"

class Module {
public:
    virtual ~Module() = default;

    // Forward computation: input -> output
    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) = 0;

    // Collect parameters like weights/biases for optimizers
    virtual std::vector<std::shared_ptr<Tensor>> parameters() const = 0;

    // Zero gradients of all parameters
    virtual void zero_grad();

    // Name (optional)
    virtual std::string name() const { return "Module"; }
};
