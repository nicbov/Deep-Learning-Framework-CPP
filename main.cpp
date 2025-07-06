#include "tensor.hpp"
#include "linear.hpp"
#include "relu_module.hpp"  
#include "mse.hpp"
#include "optimizer.hpp"
#include "sequential.hpp"
#include <iostream>
#include <vector>
#include <cmath> // for std::isnan, std::isinf

int main() {
    std::cout << "=== Initializing data ===" << std::endl;

    // Inputs: shape (2, 3)
    auto x = std::make_shared<Tensor>(std::vector<int>{2, 3}, false);
    x->data = {1, 2, 3, 4, 5, 6};

    // Targets: shape (2, 2)
    auto target = std::make_shared<Tensor>(std::vector<int>{2, 2}, false);
    target->data = {1.0, 0.0, 1.0, 0.0};

    std::cout << "=== Building model ===" << std::endl;
    auto model = std::make_shared<Sequential>();
    model->add_module(std::make_shared<Linear>(3, 4));
    model->add_module(std::make_shared<ReLU>());
    model->add_module(std::make_shared<Linear>(4, 2));

    SGD optimizer(0.01f);

    std::cout << "=== Starting training ===" << std::endl;

    for (int epoch = 0; epoch < 100; ++epoch) {
        std::cout << "\nEpoch " << epoch << std::endl;

        std::cout << "Forward pass..." << std::endl;
        auto output = model->forward(x);

        std::cout << "Computing loss..." << std::endl;
        auto loss = mse_loss(output, target);

        if (loss->data.empty()) {
            std::cerr << "❌ Loss data is empty!" << std::endl;
            break;
        }

        float loss_val = loss->data[0];
        std::cout << "Loss: " << loss_val << std::endl;

        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cerr << "❌ NaN or Inf in loss, stopping early." << std::endl;
            break;
        }

        std::cout << "Backward pass..." << std::endl;
        loss->backward();

        std::cout << "Optimizer step..." << std::endl;
        optimizer.step(model->parameters());

        std::cout << "Zeroing gradients..." << std::endl;
        model->zero_grad();
    }

    std::cout << "\n✅ Training finished.\n";
    return 0;
}
