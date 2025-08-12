/*
 * linear.cpp - fully connected linear layer implementation
 * 
 * this file implements the core linear transformation y = xW + b used in neural networks:
 * - weight initialization: he initialization for relu activations, zero bias initialization
 * - forward pass: matrix multiplication followed by bias addition with broadcasting
 * - computational graph integration: creates matmul and add operations for gradient flow
 * 
 * Why HE is important for weight initialization: it prevents vanishing gradients in deep networks
 * by scaling weights based on input dimension: std::sqrt(2.0 / in_features), which fixes the problem of 0 gradients
 * and no learning.
 */

#include "linear.hpp"
#include "ops/matmul.hpp"
#include "ops/add.hpp"
#include "ops/linear_op.hpp"  
#include "graph.hpp"

#include <random>
#include <iostream>

// global graph manager for tensor and operation lifetime management
// prevents premature destruction of weight and bias tensors during training
extern Graph global_graph;

// simple random weight initialization (currently unused but available for experimentation)
// uniform distribution between -1 and 1 for basic weight initialization
static float rand_weight() {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    return dist(-1.0f, 1.0f);
}

// xavier/glorot initialization for better gradient flow in general cases
// scales weights by std::sqrt(6.0 / (in_features + out_features)) for balanced variance
static float xavier_weight(int in_features, int out_features) {
    static std::mt19937 gen(42);
    float limit = std::sqrt(6.0f / (in_features + out_features));
    static std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

// he initialization for relu activations (optimal for deep networks)
// scales weights by std::sqrt(2.0 / in_features) to prevent vanishing gradients
// this is the recommended initialization when using relu activations
static float he_weight(int in_features) {
    static std::mt19937 gen(42);
    float limit = std::sqrt(2.0f / in_features);
    static std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

Linear::Linear(int in_features, int out_features) {
    // create weight matrix and bias vector as trainable parameters
    // these tensors will be updated by the optimizer during training
    weight = std::make_shared<Tensor>(std::vector<int>{in_features, out_features}, true);
    bias = std::make_shared<Tensor>(std::vector<int>{out_features}, true);

    // use he initialization for better performance with relu activations
    // this prevents vanishing gradients by scaling weights appropriately
    for (auto& w : weight->data) w = he_weight(in_features);
    for (auto& b : bias->data) b = 0.0f; // initialize bias to 0 for better stability

    // register parameters with global graph to prevent premature destruction
    // critical for maintaining parameter references across training epochs
    global_graph.add_tensor(weight);
    global_graph.add_tensor(bias);

    std::cout << "[Linear ctor] weight shape: ";
    for (auto d : weight->shape) std::cout << d << " ";
    std::cout << std::endl;

    std::cout << "[Linear ctor] bias shape: ";
    for (auto d : bias->shape) std::cout << d << " ";
    std::cout << std::endl;
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    // compute linear transformation: y = xW + b
    auto wx = matmul(input, weight);  // matrix multiplication: input × weight
    auto result = add(wx, bias);       // add bias (with broadcasting)

    // note: wx and result already have their creators set by matmul and add operations
    // we don't need to overwrite the creator with linear_op
    
    // add the result tensor to the global graph for lifetime management
    // this prevents premature destruction during forward pass
    global_graph.add_tensor(result);
    
    return result;
}

std::vector<std::shared_ptr<Tensor>> Linear::parameters() const {
    // return all trainable parameters for optimizer access
    // optimizer will update these tensors during training steps
    return {weight, bias};
}
