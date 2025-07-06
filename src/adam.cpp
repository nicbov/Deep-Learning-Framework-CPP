#include "adam.hpp"
#include <cmath>

Adam::Adam(float learning_rate, float beta1_, float beta2_, float epsilon_)
    : lr(learning_rate), beta1(beta1_), beta2(beta2_), epsilon(epsilon_), t(0), initialized(false) {}

void Adam::initialize_state(const std::vector<std::shared_ptr<Tensor>>& params) {
    m.clear();
    v.clear();
    m.reserve(params.size());
    v.reserve(params.size());

    for (auto& p : params) {
        m.emplace_back(p->data.size(), 0.0f);
        v.emplace_back(p->data.size(), 0.0f);
    }
    initialized = true;
}

void Adam::step(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (!initialized) {
        initialize_state(params);
    }
    ++t; // increment timestep

    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        if (!p->requires_grad) continue;

        for (size_t j = 0; j < p->data.size(); ++j) {
            float grad = p->grad[j];

            // Update biased first moment estimate
            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * grad;
            // Update biased second raw moment estimate
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * grad * grad;

            // Compute bias-corrected first moment estimate
            float m_hat = m[i][j] / (1.0f - std::pow(beta1, t));
            // Compute bias-corrected second moment estimate
            float v_hat = v[i][j] / (1.0f - std::pow(beta2, t));

            // Update parameters
            p->data[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
}

void Adam::zero_state() {
    m.clear();
    v.clear();
    t = 0;
    initialized = false;
}
