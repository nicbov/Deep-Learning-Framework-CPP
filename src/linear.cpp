#include "linear.hpp"
#include "matmul.hpp"
#include "add.hpp"
#include <random>

// Simple random initialization helper
static float rand_weight() {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(gen);
}

Linear::Linear(int in_features, int out_features) {
    weight = std::make_shared<Tensor>(std::vector<int>{in_features, out_features}, true);
    bias = std::make_shared<Tensor>(std::vector<int>{out_features}, true);

    for (auto& w : weight->data) w = rand_weight();
    for (auto& b : bias->data) b = rand_weight();
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    auto wx = matmul(input, weight);
    // Add bias by broadcasting (element-wise add along last dim)
    auto result = add(wx, bias);  
    return result;
}
