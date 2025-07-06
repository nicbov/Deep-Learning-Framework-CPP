#include "tensor.hpp"
#include "linear.hpp"
#include "relu.hpp"
#include "mse.hpp"
#include "optimizer.hpp"
#include <iostream>
#include <vector>

int main() {
    auto x = std::make_shared<Tensor>(std::vector<int>{2, 3}, false);
    x->data = {1, 2, 3, 4, 5, 6};

    auto target = std::make_shared<Tensor>(std::vector<int>{2, 2}, false);
    target->data = {1.0, 0.0, 1.0, 0.0};

    Linear layer(3, 2);
    SGD optimizer(0.01f);

    // Collect parameters for optimizer
    std::vector<std::shared_ptr<Tensor>> params = {layer.weight, layer.bias};

    for (int epoch = 0; epoch < 100; ++epoch) {
        auto z = layer.forward(x);
        auto a = relu(z);
        auto loss = mse_loss(a, target);

        loss->backward();

        optimizer.step(params);
        optimizer.zero_grad(params);

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss->data[0] << std::endl;
        }
    }

    return 0;
}
