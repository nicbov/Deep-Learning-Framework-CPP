#include "add.hpp"
#include <stdexcept>
#include "../graph.hpp"
#include "../tensor.hpp"

extern Graph global_graph;

AddOp::AddOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);
    inputs.push_back(b);
}

void AddOp::backward(Tensor& grad_output) {
    for (auto& weak_input : inputs) {
        auto input = std::const_pointer_cast<Tensor>(weak_input.lock());
        if (!input) throw std::runtime_error("AddOp: input expired");

        if (input->requires_grad) {
            if (input->grad.empty()) input->grad.resize(input->data.size(), 0.0f);
            for (size_t i = 0; i < input->grad.size(); ++i)
                input->grad[i] += grad_output.grad[i];

            if (auto c = input->creator.lock()) c->backward(*input);
        }
    }
}

std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    auto result = std::make_shared<Tensor>(a->shape, a->requires_grad || b->requires_grad);
    for (size_t i = 0; i < result->data.size(); ++i)
        result->data[i] = a->data[i] + b->data[i];

    if (result->requires_grad) {
        auto op = std::make_shared<AddOp>(a, b);
        result->creator = op;

        // Register in global graph
        global_graph.add_tensor(result);
        global_graph.add_op(op);
    }

    return result;
}
