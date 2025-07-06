#pragma once

#include <vector>
#include <memory>
#include "tensor.hpp"

class SGD {
    float lr;  // learning rate
public:
    explicit SGD(float learning_rate);
    void step(const std::vector<std::shared_ptr<Tensor>>& params);
    void zero_grad(const std::vector<std::shared_ptr<Tensor>>& params);
};
