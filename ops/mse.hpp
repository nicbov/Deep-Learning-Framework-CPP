#pragma once

#include "../tensor.hpp"
#include "../op.hpp"

class MSELossOp : public Op {
    std::shared_ptr<Tensor> target;
public:
    MSELossOp(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> pred, std::shared_ptr<Tensor> target);
