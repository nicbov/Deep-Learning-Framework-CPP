#include "linear.hpp"
#include "matmul.hpp"
#include "add.hpp"
#include <random>

static float rand_weight() {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(gen);
}

Linear::Linear(int in_features, int out_features) {
    weight = std::make_shared<Tensor>(std::vector<int>{in_features, out_features}, true);
    bias = std::make_shared<Tensor>(std::vector<int>{out_features}, true);

    // Initialize weights with random values
    for (auto& w : weight->data) w = rand_weight();

    // Initialize bias with random values
    for (auto& b : bias->data) b = rand_weight();

    // Now print shapes for debugging
    std::cout << "[Linear ctor] weight shape: ";
    for (auto d : weight->shape) std::cout << d << " ";
    std::cout << std::endl;

    std::cout << "[Linear ctor] bias shape: ";
    for (auto d : bias->shape) std::cout << d << " ";
    std::cout << std::endl;
}


std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    std::cout << "[Linear] input shape: ";
    for (auto d : input->shape) std::cout << d << " ";
    std::cout << std::endl;

    auto wx = matmul(input, weight);

    std::cout << "[Linear] wx shape: ";
    for (auto d : wx->shape) std::cout << d << " ";
    std::cout << std::endl;

    auto result = add(wx, bias);

    std::cout << "[Linear] result shape: ";
    for (auto d : result->shape) std::cout << d << " ";
    std::cout << std::endl;

    return result;
}


std::vector<std::shared_ptr<Tensor>> Linear::parameters() const {
    return {weight, bias};
}

