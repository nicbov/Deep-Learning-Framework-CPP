#include <iostream>
#include <random>
#include <cmath>

// Simple test to check weight initialization and matrix multiplication
int main() {
    std::cout << "=== Testing Numerical Stability ===" << std::endl;
    
    // Test 1: Check weight initialization (same as Linear layer)
    std::cout << "\n1. Testing weight initialization:" << std::endl;
    
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    auto rand_weight = [&]() -> float {
        return dist(gen);
    };
    
    std::vector<float> weights(36); // 9 * 4 weights
    std::vector<float> biases(4);   // 4 biases
    
    for (auto& w : weights) w = rand_weight();
    for (auto& b : biases) b = rand_weight();
    
    std::cout << "Weight values (first 10): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << weights[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Bias values: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << biases[i] << " ";
    }
    std::cout << std::endl;
    
    // Test 2: Check forward pass with small input (same as your data)
    std::cout << "\n2. Testing forward pass with small input:" << std::endl;
    
    std::vector<float> input(9, 0.1f); // Small normalized values like your data
    std::vector<float> hidden(4, 0.0f);
    std::vector<float> output(1, 0.0f);
    
    // First layer: input (1,9) @ weights (9,4) + bias (4)
    for (int i = 0; i < 4; ++i) {
        hidden[i] = biases[i];
        for (int j = 0; j < 9; ++j) {
            hidden[i] += input[j] * weights[j * 4 + i];
        }
    }
    
    std::cout << "Hidden layer values: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << hidden[i] << " ";
    }
    std::cout << std::endl;
    
    // Check if hidden values are reasonable
    bool hidden_ok = true;
    for (int i = 0; i < 4; ++i) {
        if (std::isnan(hidden[i]) || std::isinf(hidden[i])) {
            std::cout << "ERROR: Hidden layer " << i << " is NaN or Inf!" << std::endl;
            hidden_ok = false;
        } else if (std::abs(hidden[i]) > 100.0f) {
            std::cout << "WARNING: Hidden layer " << i << " is very large: " << hidden[i] << std::endl;
            hidden_ok = false;
        }
    }
    
    if (hidden_ok) {
        std::cout << "Hidden layer looks reasonable" << std::endl;
    }
    
    // Test 3: Check what happens with larger inputs
    std::cout << "\n3. Testing with larger inputs (like your actual data):" << std::endl;
    
    // Simulate some of your actual normalized data values
    std::vector<float> larger_input = {0.5f, 0.7f, 0.3f, 0.8f, 0.2f, 0.6f, 0.4f, 0.9f, 0.1f};
    
    for (int i = 0; i < 4; ++i) {
        hidden[i] = biases[i];
        for (int j = 0; j < 9; ++j) {
            hidden[i] += larger_input[j] * weights[j * 4 + i];
        }
    }
    
    std::cout << "Hidden layer with larger input: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << hidden[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
