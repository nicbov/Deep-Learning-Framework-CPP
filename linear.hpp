#pragma once

#include "src/module.hpp"
#include <memory>
#include <vector>

class Linear : public Module {
public:
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    Linear(int in_features, int out_features);

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) override;

    std::vector<std::shared_ptr<Tensor>> parameters() const override;

    std::string name() const override { return "Linear"; }

    std::vector<std::shared_ptr<Op>> ops; // store created ops

};
