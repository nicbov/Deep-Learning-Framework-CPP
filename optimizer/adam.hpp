/*
 * adam.hpp - adam optimizer for neural network training
 * 
 * implements the adam (adaptive moment estimation) optimizer
 */

#pragma once

#include "../tensor.hpp"
#include <vector>
#include <memory>

class Adam {
public:
    // constructor with default hyperparameters (good for most cases)
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    // update parameters using computed gradients
    // this is the main training step that modifies model weights
    void step(const std::vector<std::shared_ptr<Tensor>>& params);
    
    // reset optimizer state (optional, rarely needed)
    void zero_state();

private:
    // hyperparameters
    float lr;        // learning rate - controls step size
    float beta1;     // first moment decay rate (momentum)
    float beta2;     // second moment decay rate (variance)
    float epsilon;   // small constant to prevent division by zero

    // internal state
    int t;           // timestep counter for bias correction
    std::vector<std::vector<float>> m;  // first moment (momentum) for each parameter
    std::vector<std::vector<float>> v;  // second moment (variance) for each parameter

    // initialization state
    bool initialized;
    
    // initialize optimizer state for given parameters
    void initialize_state(const std::vector<std::shared_ptr<Tensor>>& params);
};
