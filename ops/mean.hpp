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
        auto input_const = inputs[0].lock();
        if (!input_const) {
            return;
        }

        if (!input_const->requires_grad) {
            return;
        }

        auto input = std::const_pointer_cast<Tensor>(input_const);

        if (input->requires_grad) {
            if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
            for (size_t i = 0; i < input->data.size(); ++i) {
                // For MSE loss, we want to scale the gradient properly
                // The factor of 2 from MSE derivative and 1/N from mean
                float grad_val = grad_output.grad[0] * 2.0f / input->data.size();
                input->grad[i] += grad_val;
            }
        }

        // CRITICAL FIX: Propagate gradients to input tensor's creator
        // But avoid infinite loops by checking if the creator is different
        auto input_creator = input->creator.lock();
        if (input_creator && input_creator.get() != this) {
            // Only propagate if the creator is different from this operation
            input_creator->backward(*input);
        }
    }
};
