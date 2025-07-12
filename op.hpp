#pragma once
#include <vector>
#include <memory>

class Tensor;

class Op {
    public:
        std::vector<std::weak_ptr<Tensor>> inputs;  
        virtual void backward(Tensor& grad_output) = 0;
        virtual ~Op() = default;
    };
    