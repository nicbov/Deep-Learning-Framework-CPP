/*
 *csv_loader.hpp - data loading and preprocessing utilities for neural network training
 */

#pragma once
#include <vector>
#include <string>

// loads housing dataset from csv file with comprehensive error handling
// returns 2d vector where each row contains 10 columns: 9 features + 1 target
// automatically skips header row and validates data integrity during loading
std::vector<std::vector<double>> load_csv(const std::string& filename);

// separates the loaded data into features and targets for neural network training
// features: first 8 columns plus ocean_proximity (9 total features for housing prediction)
// targets: median_house_value column used as the regression target during training
void split_features_targets(
    const std::vector<std::vector<double>>& data,
    std::vector<std::vector<double>>& features,
    std::vector<std::vector<double>>& targets
);
