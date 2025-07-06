#pragma once

#include "tensor.hpp"
#include "op.hpp"

class ReLUOp : public Op {
public:
    ReLUOp(std::shared_ptr<Tensor> input);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> input);
