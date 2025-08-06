#pragma once
#include "../op.hpp"
#include <memory>

class PowOp : public Op {
public:
    float exponent;
    PowOp(const std::shared_ptr<Tensor>& input, float exponent_);
    std::shared_ptr<Tensor> input_keep_alive;

    void backward(Tensor& grad_output) override;
};
