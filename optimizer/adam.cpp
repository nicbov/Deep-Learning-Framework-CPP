#include "adam.hpp"
#include <cmath>
#include <iostream>

Adam::Adam(float learning_rate, float beta1_, float beta2_, float epsilon_)
    : lr(learning_rate), beta1(beta1_), beta2(beta2_), epsilon(epsilon_), t(0), initialized(false) {}

void Adam::initialize_state(const std::vector<std::shared_ptr<Tensor>>& params) {
    std::cout << "[Adam] Initializing state for " << params.size() << " parameters\n";
    m.clear();
    v.clear();
    m.reserve(params.size());
    v.reserve(params.size());

    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        size_t size = p->data.size();
        std::cout << "[Adam] Param " << i << " size: " << size << "\n";
        m.emplace_back(size, 0.0f);
        v.emplace_back(size, 0.0f);
    }
    initialized = true;
    std::cout << "[Adam] Initialization done.\n";
}

void Adam::step(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (!initialized) {
        initialize_state(params);
    }
    ++t; // increment timestep
    
    // Gradient clipping to prevent explosion
    float max_grad_norm = 1.0f;  // Clip gradients to this max norm
    for (auto& param : params) {
        if (!param->requires_grad || param->grad.empty()) continue;
        for (auto& grad : param->grad) {
            if (std::abs(grad) > max_grad_norm) {
                grad = std::copysign(max_grad_norm, grad);
            }
        }
    }
    std::cout << "[Adam] Step " << t << " updating parameters.\n";

    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        if (!p->requires_grad) {
            std::cout << "[Adam] Param " << i << " does not require grad, skipping.\n";
            continue;
        }

        if (p->grad.size() != p->data.size()) {
            std::cerr << "[Adam] ERROR: Grad and data size mismatch for param " << i
                      << " grad size: " << p->grad.size() << ", data size: " << p->data.size() << "\n";
            continue;
        }

        std::cout << "[Adam] Updating param " << i << " (size " << p->data.size() << ")\n";
        for (size_t j = 0; j < p->data.size(); ++j) {
            float grad = p->grad[j];

            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * grad;
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * grad * grad;

            float m_hat = m[i][j] / (1.0f - std::pow(beta1, t));
            float v_hat = v[i][j] / (1.0f - std::pow(beta2, t));

            float update = lr * m_hat / (std::sqrt(v_hat) + epsilon);
            p->data[j] -= update;

            // Print first few updates for debug
            if (j < 3) {
                std::cout << "  idx " << j << ": grad=" << grad << ", m=" << m[i][j] << ", v=" << v[i][j]
                          << ", m_hat=" << m_hat << ", v_hat=" << v_hat << ", update=" << update
                          << ", new_param=" << p->data[j] << "\n";
            }
        }
    }
    std::cout << "[Adam] Step " << t << " complete.\n";
}

void Adam::zero_state() {
    std::cout << "[Adam] Clearing optimizer state.\n";
    m.clear();
    v.clear();
    t = 0;
    initialized = false;
}
