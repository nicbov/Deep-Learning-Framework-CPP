#include "linear.hpp"
#include "ops/matmul.hpp"
#include "ops/add.hpp"
#include "ops/linear_op.hpp"  
#include <random>

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

    if (result->requires_grad) {
        auto op = std::make_shared<LinearOp>(input, weight, bias);
        result->creator = op;
        ops.push_back(op);  // keep op alive here
    }

    return result;
}


std::vector<std::shared_ptr<Tensor>> Linear::parameters() const {
    return {weight, bias};
}
