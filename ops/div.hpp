#pragma once
#include "../op.hpp"

class DivOp : public Op {
public:
    float scalar;
    DivOp(std::shared_ptr<Tensor> input, float scalar_);
    void backward(Tensor& grad_output) override;
};
