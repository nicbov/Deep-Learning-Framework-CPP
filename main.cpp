#include "tensor.hpp"
#include "linear.hpp"
#include "ops/mse.hpp"
#include "model/sequential.hpp"
#include "optimizer/adam.hpp"
#include "data/csv_loader.hpp" 
#include <iostream>
#include <vector>
#include <cmath>


int main() {
    std::cout << "=== Loading CSV data ===" << std::endl;

    auto data = load_csv("data/final_training_data.csv");

    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> targets;

    split_features_targets(data, features, targets);

    const int sample_count = features.size();
    const int input_dim = 3;
    const int output_dim = 2;

    std::cout << "Loaded " << sample_count << " samples." << std::endl;

    // ✅ Set requires_grad = true for input so graph links are created
    auto x = std::make_shared<Tensor>(std::vector<int>{sample_count, input_dim}, true);
    auto target = std::make_shared<Tensor>(std::vector<int>{sample_count, output_dim}, false);

    for (int i = 0; i < sample_count; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            x->data[i * input_dim + j] = static_cast<float>(features[i][j]);
        }
        for (int j = 0; j < output_dim; ++j) {
            target->data[i * output_dim + j] = static_cast<float>(targets[i][j]);
        }
    }

    std::cout << "=== Building model ===" << std::endl;

    auto model = std::make_shared<Sequential>();
    model->add_module(std::make_shared<Linear>(input_dim, 4));
    model->add_module(std::make_shared<Linear>(4, output_dim));

    Adam optimizer(0.001f);

    std::cout << "=== Starting training ===" << std::endl;

    for (int epoch = 0; epoch < 100; ++epoch) {
        std::cout << "\nEpoch " << epoch << std::endl;

        auto output = model->forward(x);
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

        loss->backward();

        // print a sample gradient to confirm it’s flowing
        auto params = model->parameters();
        if (!params.empty() && !params[0]->grad.empty()) {
            std::cout << "[Debug] First param grad[0]: " << params[0]->grad[0] << std::endl;
        }

        optimizer.step(model->parameters());

        model->zero_grad();
    }

    std::cout << "\n✅ Training finished.\n";
    return 0;
}
