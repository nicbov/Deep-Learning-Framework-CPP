#pragma once

#include "tensor.hpp"
#include "op.hpp"

class MulOp : public Op {
public:
    MulOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
