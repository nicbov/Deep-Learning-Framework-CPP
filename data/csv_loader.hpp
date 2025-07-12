#pragma once
#include <vector>
#include <string>

std::vector<std::vector<double>> load_csv(const std::string& filename);

void split_features_targets(
    const std::vector<std::vector<double>>& data,
    std::vector<std::vector<double>>& features,
    std::vector<std::vector<double>>& targets
);
