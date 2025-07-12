#pragma once

#include "../tensor.hpp"
#include <vector>
#include <memory>

class Adam {
public:
    Adam(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    void step(const std::vector<std::shared_ptr<Tensor>>& params);
    void zero_state(); // resets moment vectors (optional)

private:
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    int t; // timestep

    // Store moments for each param (same size as params vector)
    std::vector<std::vector<float>> m; // first moment
    std::vector<std::vector<float>> v; // second moment

    bool initialized;
    void initialize_state(const std::vector<std::shared_ptr<Tensor>>& params);
};
