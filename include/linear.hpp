#pragma once

#include "tensor.hpp"
#include <memory>

class Linear {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features);
    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input);
};
