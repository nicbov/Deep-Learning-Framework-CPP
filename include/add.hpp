#pragma once

#include "tensor.hpp"
#include "op.hpp"

class AddOp : public Op {
public:
    AddOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
