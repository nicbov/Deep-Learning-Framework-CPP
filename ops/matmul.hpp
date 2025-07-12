#pragma once

#include "../tensor.hpp"
#include "../op.hpp"

class MatMulOp : public Op {
public:
    MatMulOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
