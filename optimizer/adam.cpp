/*
 * adam.cpp - adam optimizer implementation for neural network training
 * 
 * this optimizer implements the adam (adaptive moment estimation) algorithm:
 * - combines momentum (first moment) and adaptive learning rates (second moment)
 * - automatically adjusts learning rates for each parameter based on gradient history
 * - provides stable training for deep neural networks with minimal hyperparameter tuning
 * - includes gradient clipping to prevent training instability and exploding gradients
 */

#include "adam.hpp"
#include <cmath>
#include <iostream>

// constructor initializes hyperparameters with sensible defaults
// beta1=0.9 provides momentum, beta2=0.999 provides adaptive learning rate scaling
Adam::Adam(float learning_rate, float beta1_, float beta2_, float epsilon_)
    : lr(learning_rate), beta1(beta1_), beta2(beta2_), epsilon(epsilon_), t(0), initialized(false) {}

// initializes momentum and variance buffers for all parameters
// these buffers will store running averages of gradients and squared gradients
// called automatically on first step if not manually initialized
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
        // initialize momentum buffer (first moment) for parameter i
        m.emplace_back(size, 0.0f);
        // initialize variance buffer (the second moment) for parameter i
        v.emplace_back(size, 0.0f);
    }
    initialized = true;
    std::cout << "[Adam] Initialization done.\n";
}

// performs one optimization step using the adam algorithm
// updates all parameters using their computed gradients and stored momentum/variance
void Adam::step(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (!initialized) {
        initialize_state(params);
    }
    ++t; // increment timestep for bias correction
    
    // gradient clipping prevents extreme gradient values that could destabilize training
    // clips gradients to maximum norm of 1.0 to maintain numerical stability
    float max_grad_norm = 1.0f;  // clip gradients to this max norm
    for (auto& param : params) {
        if (!param->requires_grad || param->grad.empty()) continue;
        for (auto& grad : param->grad) {
            if (std::abs(grad) > max_grad_norm) {
                grad = std::copysign(max_grad_norm, grad);
            }
        }
    }
    std::cout << "[Adam] Step " << t << " updating parameters.\n";

    // update each parameter using adam algorithm
    for (size_t i = 0; i < params.size(); ++i) {
        auto& p = params[i];
        if (!p->requires_grad) {
            std::cout << "[Adam] Param " << i << " does not require grad, skipping.\n";
            continue;
        }

        // validate gradient and parameter size consistency
        if (p->grad.size() != p->data.size()) {
            std::cerr << "[Adam] ERROR: Grad and data size mismatch for param " << i
                      << " grad size: " << p->grad.size() << ", data size: " << p->data.size() << "\n";
            continue;
        }

        std::cout << "[Adam] Updating param " << i << " (size " << p->data.size() << ")\n";
        for (size_t j = 0; j < p->data.size(); ++j) {
            float grad = p->grad[j];

            // update momentum (first moment) - exponential moving average of gradients
            m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * grad;
            // update variance (second moment) - exponential moving average of squared gradients
            v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * grad * grad;

            // bias correction for early training steps when moving averages are biased toward zero
            float m_hat = m[i][j] / (1.0f - std::pow(beta1, t));
            float v_hat = v[i][j] / (1.0f - std::pow(beta2, t));

            // compute adaptive learning rate and parameter update
            float update = lr * m_hat / (std::sqrt(v_hat) + epsilon);
            p->data[j] -= update;

            // print first few updates for debugging and monitoring training progress
            if (j < 3) {
                std::cout << "  idx " << j << ": grad=" << grad << ", m=" << m[i][j] << ", v=" << v[i][j]
                          << ", m_hat=" << m_hat << ", v_hat=" << v_hat << ", update=" << update
                          << ", new_param=" << p->data[j] << "\n";
            }
        }
    }
    std::cout << "[Adam] Step " << t << " complete.\n";
}

// clears all optimizer state including momentum and variance buffers
// useful for restarting training or switching between different optimization strategies
void Adam::zero_state() {
    std::cout << "[Adam] Clearing optimizer state.\n";
    m.clear();
    v.clear();
    t = 0;
    initialized = false;
}
