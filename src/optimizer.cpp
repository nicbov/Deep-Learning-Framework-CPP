#include "optimizer.hpp"

SGD::SGD(float learning_rate) : lr(learning_rate) {}

void SGD::step(const std::vector<std::shared_ptr<Tensor>>& params) {
    for (auto& p : params) {
        if (!p->requires_grad) continue;
        for (size_t i = 0; i < p->data.size(); ++i) {
            p->data[i] -= lr * p->grad[i];
        }
    }
}

void SGD::zero_grad(const std::vector<std::shared_ptr<Tensor>>& params) {
    for (auto& p : params) {
        if (p->requires_grad) {
            p->zero_grad();
        }
    }
}
