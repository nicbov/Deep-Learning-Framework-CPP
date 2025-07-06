#pragma once

#include "tensor.hpp"
#include "op.hpp"
#include <memory>

class TanhOp : public Op {
public:
    TanhOp(std::shared_ptr<Tensor> input);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> tanh(std::shared_ptr<Tensor> input);
