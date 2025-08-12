/*
 * main.cpp - neural network training script for housing price prediction
 * 
 * this file orchestrates the complete training pipeline including:
 * - data loading and preprocessing (csv parsing, normalization)
 * - model construction (sequential neural network with linear + relu layers)
 * - training loop with adam optimizer and early stopping
 * - gradient monitoring and debugging output
 * - prediction tracking and final model evaluation
 * 
 * architecture: 9 input features -> 16 -> 8 -> 4 -> 1 output (housing price)
 * uses min-max normalization for both features and targets to ensure stable training
 */

#include "tensor.hpp"
#include "linear.hpp"
#include "relu.hpp"

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
#include <limits>

// global computation graph manager - keeps all tensors and operations alive during training
// critical for preventing premature destruction of intermediate computation results
Graph global_graph;  

// denormalize housing prices back to original dollar amounts for human-readable output
// normalization range: $14,999 to $500,001 (california housing market extremes)
float denormalize_price(float normalized_value) {
    float min_val = 14999.0f;   
    float max_val = 500001.0f;  
    return normalized_value * (max_val - min_val) + min_val;
}

// tracks prediction accuracy across training epochs for model evaluation
// stores (predicted, target) pairs and loss values for each epoch
struct PredictionRecord {
    int epoch;
    std::vector<std::pair<float, float>> predictions;  // (normalized_pred_price, normalized_target_price)
    float loss;
};

std::vector<PredictionRecord> prediction_history;

// monitors gradient flow to detect training issues
// only warns if all gradients are zero (which would prevent learning)
void print_gradient_stats(const std::vector<std::shared_ptr<Tensor>>& params) {
    if (params.empty()) return;
    
    bool has_nonzero_grads = false;
    for (auto& param : params) {
        if (!param->requires_grad || param->grad.empty()) continue;
        for (auto& grad : param->grad) {
            if (std::abs(grad) > 1e-8f) {
                has_nonzero_grads = true;
                break;
            }
        }
        if (has_nonzero_grads) break;
    }
    
    if (!has_nonzero_grads) {
        std::cout << "[WARNING] All gradients are zero! This will prevent learning." << std::endl;
    }
}

// tracks parameter value changes across training to monitor convergence
// stores first few values of each parameter for debugging
void track_parameter_changes(const std::vector<std::shared_ptr<Tensor>>& params, 
                           std::vector<std::vector<float>>& param_history) {
    if (param_history.empty()) {
        param_history.resize(params.size());
    }
    
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->data.empty()) continue;
        
        // store first few values of each parameter
        for (int j = 0; j < std::min(3, (int)params[i]->data.size()); ++j) {
            param_history[i].push_back(params[i]->data[j]);
        }
    }
}

int main() {
    std::cout << "=== Loading CSV data ===" << std::endl;

    auto data = load_csv("data/housing_clean.csv");

    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> targets;

    split_features_targets(data, features, targets);

    const int sample_count = features.size();
    const int input_dim = 9;    // 9 features: longitude, latitude, age, rooms, bedrooms, population, households, income, ocean_proximity
    const int output_dim = 1;   // 1 target: median_house_value

    std::cout << "Loaded " << sample_count << " samples." << std::endl;
    
    // validate data integrity before proceeding
    if (sample_count == 0) {
        std::cerr << "Error: No data loaded!" << std::endl;
        return 1;
    }
    
    std::cout << "First sample features: ";
    for (int j = 0; j < std::min(5, input_dim); ++j) {
        std::cout << features[0][j] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First sample target: " << targets[0][0] << std::endl;
    std::cout << "Last sample target: " << targets[sample_count-1][0] << std::endl;

    // hardcoded normalization parameters for california housing dataset
    // these values were computed from the full dataset to ensure consistent scaling
    const float feature_mins[9] = {
        -124.35f, 32.54f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 0.4999f, -1.0f
    };
    const float feature_maxs[9] = {
        -114.31f, 41.95f, 52.0f, 39320.0f, 6445.0f, 35682.0f, 6082.0f, 15.0001f, 2.0f
    };

    // target normalization range for median_house_value
    const float target_min = 14999.0f;
    const float target_max = 500001.0f;

    // create input and target tensors with proper gradient tracking
    // input tensor requires gradients for backpropagation, target does not
    auto x = std::make_shared<Tensor>(std::vector<int>{sample_count, input_dim}, true);
    auto target = std::make_shared<Tensor>(std::vector<int>{sample_count, output_dim}, false);

    global_graph.add_tensor(x);
    global_graph.add_tensor(target);

    // normalize all features and targets to [0,1] range for stable training
    // this prevents gradient explosion and ensures consistent learning rates
    for (int i = 0; i < sample_count; ++i) {
        for (int j = 0; j < input_dim; ++j) {
            float raw_val = static_cast<float>(features[i][j]);
            float norm_val = (raw_val - feature_mins[j]) / (feature_maxs[j] - feature_mins[j]);
            x->data[i * input_dim + j] = norm_val;
        }
        // normalize target
        float raw_target = static_cast<float>(targets[i][0]);
        float norm_target = (raw_target - target_min) / (target_max - target_min);
        target->data[i * output_dim + 0] = norm_target;
    }
    
    // validate normalization results
    std::cout << "Normalization validation:" << std::endl;
    std::cout << "First sample normalized features: ";
    for (int j = 0; j < std::min(5, input_dim); ++j) {
        std::cout << x->data[j] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "First sample normalized target: " << target->data[0] << std::endl;
    std::cout << "Target range check - min: " << target_min << ", max: " << target_max << std::endl;

    std::cout << "=== Building model ===" << std::endl;
    auto model = std::make_shared<Sequential>();
    model->add_module(std::make_shared<Linear>(input_dim, 16));  // first hidden layer: 9 -> 16
    model->add_module(std::make_shared<ReLU>());                 // activation after first layer
    model->add_module(std::make_shared<Linear>(16, 8));         // second hidden layer: 16 -> 8
    model->add_module(std::make_shared<ReLU>());                 // activation after second layer
    model->add_module(std::make_shared<Linear>(8, 4));          // third hidden layer: 8 -> 4
    model->add_module(std::make_shared<ReLU>());                 // activation after third layer
    model->add_module(std::make_shared<Linear>(4, output_dim)); // output layer: 4 -> 1
    // no activation on output layer for regression (linear output)

    // adam optimizer with increased learning rate for faster convergence
    // beta1=0.9, beta2=0.999 provide good momentum and adaptive learning
    Adam optimizer(0.01f);

    std::cout << "=== Starting training ===" << std::endl;
    
    float best_loss = std::numeric_limits<float>::infinity();
    int patience = 20;
    int no_improvement = 0;
    
    // track parameter changes for debugging and convergence analysis
    std::vector<std::vector<float>> param_history;

    for (int epoch = 0; epoch < 200; ++epoch) {
        std::cout << "\nEpoch " << epoch << std::endl;

        // zero gradients before forward pass to prevent gradient accumulation
        // this is critical for proper backpropagation
        model->zero_grad();

        auto output = model->forward(x);
        
        // clamp output to prevent extreme values that could destabilize training
        // allows some overflow (up to 2.0) for learning, but prevents nan/inf
        for (auto& val : output->data) {
            if (std::isnan(val) || std::isinf(val)) {
                val = 0.5f; // default to middle of range if nan/inf
            } else if (val < -1.0f) {
                val = -1.0f; // clamp to reasonable range
            } else if (val > 2.0f) {
                val = 2.0f; // allow some overflow for learning
            }
        }
        
        // display predictions every 10 epochs to monitor training progress
        if (epoch % 10 == 0) {  
            std::cout << "\n--- Predictions vs Targets (first 5 samples) ---" << std::endl;
            std::cout << std::fixed << std::setprecision(2);

            std::vector<std::pair<float, float>> epoch_records;

            for (int i = 0; i < std::min(5, sample_count); ++i) {
                float pred_price_norm = output->data[i * output_dim + 0];
                float target_price_norm = target->data[i * output_dim + 0];

                float pred_price = denormalize_price(pred_price_norm);
                float target_price = denormalize_price(target_price_norm);

                std::cout << "Sample " << i << ": "
                          << "Predicted Price = $" << pred_price 
                          << " | Target Price = $" << target_price << std::endl;

                epoch_records.push_back({pred_price_norm, target_price_norm});
            }
            prediction_history.push_back({epoch, epoch_records, 0.0f});
            std::cout << "-----------------------------------------------\n" << std::endl;
        }

        // compute mean squared error loss for regression
        auto loss = mse_loss(output, target);

        if (loss->data.empty()) {
            std::cerr << "Loss data is empty!" << std::endl;
            break;
        }

        float loss_val = loss->data[0];
        std::cout << "Loss: " << loss_val << std::endl;

        if (!prediction_history.empty() && epoch % 10 == 0) {
            prediction_history.back().loss = loss_val;
        }

        // early stopping on nan/inf to prevent training instability
        if (std::isnan(loss_val) || std::isinf(loss_val)) {
            std::cerr << "NaN or Inf in loss, stopping early." << std::endl;
            break;
        }

        // early stopping: stop if no improvement for 20 epochs
        // prevents overfitting and saves computation time
        if (loss_val < best_loss) {
            best_loss = loss_val;
            no_improvement = 0;
            std::cout << "New best loss: " << best_loss << std::endl;
        } else {
            no_improvement++;
            if (no_improvement >= patience) {
                std::cout << "Early stopping after " << patience << " epochs without improvement" << std::endl;
                break;
            }
        }

        // backpropagate gradients through the computation graph
        loss->backward();

        // monitor gradient flow to ensure proper learning
        auto params = model->parameters();
        print_gradient_stats(params);
        
        // debug: verify gradients are actually flowing through the network
        std::cout << "[Debug] After backward pass:" << std::endl;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i]->requires_grad && !params[i]->grad.empty()) {
                float max_grad = *std::max_element(params[i]->grad.begin(), params[i]->grad.end(), 
                    [](float a, float b) { return std::abs(a) < std::abs(b); });
                std::cout << "  Param " << i << " max grad: " << max_grad << std::endl;
            }
        }

        // gradient clipping prevents gradient explosion in deep networks
        // clips gradients to maximum norm of 1.0 for stability
        float max_grad_norm = 1.0f;
        for (auto& param : params) {
            if (!param->requires_grad || param->grad.empty()) continue;
            for (auto& grad : param->grad) {
                if (std::abs(grad) > max_grad_norm) {
                    grad = std::copysign(max_grad_norm, grad);
                }
            }
        }

        // additional gradient debugging and extreme value detection
        if (!params.empty() && !params[0]->grad.empty()) {
            std::cout << "[Debug] First param grad[0]: " << params[0]->grad[0] << std::endl;
            
            // check for extreme gradient values that could destabilize training
            bool extreme_grads = false;
            for (auto& param : params) {
                if (!param->requires_grad || param->grad.empty()) continue;
                for (auto& grad : param->grad) {
                    if (std::isnan(grad) || std::isinf(grad) || std::abs(grad) > 1000.0f) {
                        std::cout << "[Debug] Extreme gradient detected: " << grad << std::endl;
                        extreme_grads = true;
                    }
                }
            }
            if (extreme_grads) {
                std::cout << "[Debug] Extreme gradients detected, training may be unstable" << std::endl;
            }
        }

        // track parameter changes for convergence analysis
        track_parameter_changes(model->parameters(), param_history);
        
        // update parameters using adam optimizer
        optimizer.step(model->parameters());
        
        // clear computational graph after optimizer step to free memory
        // this must happen after optimizer step, not before, to preserve gradients
        global_graph.clear();
    }

    // final prediction summary showing model performance across all epochs
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🏠 FINAL PREDICTION SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (const auto& record : prediction_history) {
        std::cout << "\n📊 EPOCH " << record.epoch << " (Loss: " << record.loss << ")" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        
        for (size_t i = 0; i < record.predictions.size(); ++i) {
            float pred_price = denormalize_price(record.predictions[i].first);
            float target_price = denormalize_price(record.predictions[i].second);

            std::cout << "Sample " << i << ": "
                      << "Predicted Price = $" << std::fixed << std::setprecision(2) << pred_price
                      << " | Target Price = $" << target_price << std::endl;
        }
    }
    
    // parameter change summary showing how weights evolved during training
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "🔧 PARAMETER CHANGE SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    for (size_t i = 0; i < param_history.size(); ++i) {
        if (param_history[i].size() < 6) continue; // need at least 2 epochs
        
        std::cout << "\nParameter " << i << " changes:" << std::endl;
        std::cout << "  Initial values: ";
        for (int j = 0; j < std::min(3, (int)param_history[i].size()/2); ++j) {
            std::cout << param_history[i][j] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "  Final values:  ";
        for (int j = param_history[i].size() - 3; j < (int)param_history[i].size(); ++j) {
            if (j >= 0) std::cout << param_history[i][j] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "✅ Training finished.\n";
    return 0;
}
