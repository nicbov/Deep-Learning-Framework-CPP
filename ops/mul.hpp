#pragma once
#include "../tensor.hpp"
#include "../op.hpp"
#include "../tensor_ops.hpp"
#include "../graph.hpp"

extern Graph global_graph;

class MulOp : public Op {
    std::shared_ptr<Tensor> a_keep_alive;
    std::shared_ptr<Tensor> b_keep_alive;

public:
    MulOp(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
        auto a_nc = std::const_pointer_cast<Tensor>(a);
        auto b_nc = std::const_pointer_cast<Tensor>(b);

        inputs.push_back(a_nc);
        inputs.push_back(b_nc);

        a_keep_alive = a_nc;
        b_keep_alive = b_nc;
    }

    void backward(Tensor& grad_output) override {
        auto a_const = inputs[0].lock();
        auto b_const = inputs[1].lock();
        if (!a_const || !b_const) return;

        auto a = std::const_pointer_cast<Tensor>(a_const);
        auto b = std::const_pointer_cast<Tensor>(b_const);

        if (a->requires_grad) {
            if (a->grad.empty()) a->grad.resize(a->data.size(), 0.0f);
            for (size_t i = 0; i < a->grad.size(); ++i)
                a->grad[i] += grad_output.grad[i] * b->data[i];
            if (auto c = a->creator.lock()) c->backward(*a);
        }

        if (b->requires_grad) {
            if (b->grad.empty()) b->grad.resize(b->data.size(), 0.0f);
            for (size_t i = 0; i < b->grad.size(); ++i)
                b->grad[i] += grad_output.grad[i] * a->data[i];
            if (auto c = b->creator.lock()) c->backward(*b);
        }
    }
};

