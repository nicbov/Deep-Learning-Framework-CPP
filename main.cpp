#include "tensor.hpp"
#include "linear.hpp"
#include "ops/mse.hpp"
#include "model/sequential.hpp"
#include "optimizer/adam.hpp"
#include "data/csv_loader.hpp"
#include "graph.hpp"  

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <iomanip>

Graph global_graph;  

// Denormalization functions
float denormalize_close(float normalized_value) {
    // From norm_params.json: "Next_Close": {"min": 2.39, "max": 7.78}
    float min_val = 2.39f;
    float max_val = 7.78f;
    return normalized_value * (max_val - min_val) + min_val;
}

float denormalize_volume(float normalized_value) {
    // From norm_params.json: "Next_Volume": {"min": 19270100.0, "max": 375003100.0}
    float min_val = 19270100.0f;
    float max_val = 375003100.0f;
    return normalized_value * (max_val - min_val) + min_val;
}

// Structure to store prediction history
struct PredictionRecord {
    int epoch;
    std::vector<std::pair<float, float>> predictions;  // (pred_close, pred_volume)
    std::vector<std::pair<float, float>> targets;      // (target_close, target_volume)
    float loss;
};

std::vector<PredictionRecord> prediction_history;

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

    global_graph.add_tensor(x);      // ✅ 3. register x
    global_graph.add_tensor(target); // ✅ 3. register target

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

    Adam optimizer(0.0001f);  // Reduced learning rate for stability

    std::cout << "=== Starting training ===" << std::endl;

    for (int epoch = 0; epoch < 100; ++epoch) {
        std::cout << "\nEpoch " << epoch << std::endl;

        auto output = model->forward(x);
        
        // Display predictions vs targets for first 5 samples
        if (epoch % 10 == 0) {  // Show every 10 epochs to avoid spam
            std::cout << "\n--- Predictions vs Targets (first 5 samples) ---" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            
            std::vector<std::pair<float, float>> epoch_predictions;
            std::vector<std::pair<float, float>> epoch_targets;
            
            for (int i = 0; i < std::min(5, sample_count); ++i) {
                float pred_close = output->data[i * output_dim + 0];
                float pred_volume = output->data[i * output_dim + 1];
                float target_close = target->data[i * output_dim + 0];
                float target_volume = target->data[i * output_dim + 1];
                
                // Denormalize values for display
                float denorm_pred_close = denormalize_close(pred_close);
                float denorm_pred_volume = denormalize_volume(pred_volume);
                float denorm_target_close = denormalize_close(target_close);
                float denorm_target_volume = denormalize_volume(target_volume);
                
                std::cout << "Sample " << i << ":" << std::endl;
                std::cout << "  Prediction: Close=$" << denorm_pred_close 
                         << ", Volume=" << std::fixed << std::setprecision(0) << denorm_pred_volume << std::endl;
                std::cout << "  Target:     Close=$" << denorm_target_close 
                         << ", Volume=" << denorm_target_volume << std::endl;
                std::cout << std::setprecision(4);
                
                epoch_predictions.push_back({pred_close, pred_volume});
                epoch_targets.push_back({target_close, target_volume});
            }
            std::cout << "-----------------------------------------------\n" << std::endl;
            
            // Store for final summary
            prediction_history.push_back({epoch, epoch_predictions, epoch_targets, 0.0f});
        }
        
        auto loss = mse_loss(output, target);

        if (loss->data.empty()) {
            std::cerr << "❌ Loss data is empty!" << std::endl;
            break;
        }

        float loss_val = loss->data[0];
        std::cout << "Loss: " << loss_val << std::endl;
        
        // Update loss in prediction history
        if (!prediction_history.empty() && epoch % 10 == 0) {
            prediction_history.back().loss = loss_val;
        }

        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cerr << "❌ NaN or Inf in loss, stopping early." << std::endl;
            break;
        }

        loss->backward();

        // print a sample gradient to confirm it's flowing
        auto params = model->parameters();
        if (!params.empty() && !params[0]->grad.empty()) {
            std::cout << "[Debug] First param grad[0]: " << params[0]->grad[0] << std::endl;
        }

        optimizer.step(model->parameters());
        model->zero_grad();
    }

    // Final prediction summary
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🎯 FINAL PREDICTION SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (const auto& record : prediction_history) {
        std::cout << "\n📊 EPOCH " << record.epoch << " (Loss: " << record.loss << ")" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < record.predictions.size(); ++i) {
            float pred_close = record.predictions[i].first;
            float pred_volume = record.predictions[i].second;
            float target_close = record.targets[i].first;
            float target_volume = record.targets[i].second;
            
            // Denormalize values
            float denorm_pred_close = denormalize_close(pred_close);
            float denorm_pred_volume = denormalize_volume(pred_volume);
            float denorm_target_close = denormalize_close(target_close);
            float denorm_target_volume = denormalize_volume(target_volume);
            
            std::cout << "Sample " << i << ":" << std::endl;
            std::cout << "  📈 Close: $" << std::fixed << std::setprecision(2) << denorm_pred_close 
                     << " (pred) vs $" << denorm_target_close << " (target)" << std::endl;
            std::cout << "  📊 Volume: " << std::fixed << std::setprecision(0) << denorm_pred_volume 
                     << " (pred) vs " << denorm_target_volume << " (target)" << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "✅ Training finished.\n";
    return 0;
}
