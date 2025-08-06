#include "csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath> // For std::isnan, std::isinf

std::vector<std::vector<double>> load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return data;
    }

    std::cout << "Successfully opened file: " << filename << std::endl;
    std::getline(file, line); // Skip header

    int line_num = 1; // for error messages

    while (std::getline(file, line)) {
        ++line_num;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        bool invalid_row = false;

        while (std::getline(ss, cell, ',')) {
            try {
                double val = std::stod(cell);
                if (std::isnan(val) || std::isinf(val)) {
                    std::cerr << "Invalid value (NaN or Inf) at line " << line_num << ": " << val << std::endl;
                    invalid_row = true;
                    break;
                }
                row.push_back(val);
            } catch (const std::exception& e) {
                std::cerr << "Conversion error at line " << line_num << ": " << e.what() << std::endl;
                invalid_row = true;
                break;
            }
        }

        if (!invalid_row) {
            if (row.size() == 5) {
                data.push_back(row);
            } else {
                std::cerr << "Warning: Skipping line " << line_num << " due to unexpected number of columns: " << row.size() << std::endl;
            }
        }
    }

    std::cout << "Loaded " << data.size() << " rows from CSV" << std::endl;
    return data;
}

void split_features_targets(
    const std::vector<std::vector<double>>& data,
    std::vector<std::vector<double>>& features,
    std::vector<std::vector<double>>& targets
) {
    for (const auto& row : data) {
        features.push_back({ row[0], row[1], row[2] });
        targets.push_back({ row[3], row[4] });
    }
}
