#pragma once
#include "../tensor.hpp"
#include "../op.hpp"
#include "../graph.hpp"  

extern Graph global_graph;

class MeanOp : public Op {
    int count;

public:
    MeanOp(const std::shared_ptr<Tensor>& input, int count_) : count(count_) {
        inputs.push_back(std::const_pointer_cast<Tensor>(input));
    }

    void backward(Tensor& grad_output) override {
        std::cout << "[MeanOp::backward] Called\n";

        auto input_const = inputs[0].lock();
        if (!input_const) {
            std::cout << "[MeanOp::backward] input expired\n";
            return;
        }

        if (!input_const->requires_grad) {
            std::cout << "[MeanOp::backward] input does not require grad\n";
            return;
        }

        auto input = std::const_pointer_cast<Tensor>(input_const);

        std::cout << "[MeanOp::backward] grad_output.grad.size() = " << grad_output.grad.size() << "\n";
        if (!grad_output.grad.empty()) {
            std::cout << "[MeanOp::backward] grad_output.grad[0] = " << grad_output.grad[0] << "\n";
        }

        std::cout << "[MeanOp::backward] input data size = " << input->data.size() << ", count = " << count << "\n";

        if (input->grad.empty()) {
            input->grad.resize(input->data.size(), 0.0f);
            std::cout << "[MeanOp::backward] grad buffer initialized for input\n";
        }

        float grad_val = grad_output.grad[0] / count;
        std::cout << "[MeanOp::backward] grad_val = " << grad_val << "\n";

        for (size_t i = 0; i < input->grad.size(); ++i) {
            input->grad[i] += grad_val;
        }
        std::cout << "[MeanOp::backward] input->grad updated\n";

        if (auto c = input->creator.lock()) {
            std::cout << "[MeanOp::backward] Recursing into creator\n";
            c->backward(*input);
        } else {
            std::cout << "[MeanOp::backward] No creator to recurse into\n";
        }
    }

};
