#pragma once

#include "tensor.hpp"
#include "op.hpp"
#include <memory>

class SigmoidOp : public Op {
public:
    SigmoidOp(std::shared_ptr<Tensor> input);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> sigmoid(std::shared_ptr<Tensor> input);
