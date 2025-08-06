#pragma once

#include <memory>
#include "../tensor.hpp"
#include "../op.hpp"

class Tensor;

class LinearOp : public Op {
public:
    std::shared_ptr<Tensor> input;  
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;

    LinearOp(const std::shared_ptr<Tensor>& input_,
             const std::shared_ptr<Tensor>& weight_,
             const std::shared_ptr<Tensor>& bias_);

    void backward(Tensor& grad_output) override;
};
