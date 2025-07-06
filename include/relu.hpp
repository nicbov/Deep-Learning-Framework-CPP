#pragma once

#include "tensor.hpp"
#include "op.hpp"
#include <memory>
#include <vector>

class ReLUOp : public Op {
public:
    // Change inputs to weak_ptr to break cycle
    std::vector<std::weak_ptr<Tensor>> inputs;

    ReLUOp(std::shared_ptr<Tensor> input);

    void backward(Tensor& grad_output) override;
};

std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> input);
