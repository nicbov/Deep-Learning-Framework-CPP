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
        int column_count = 0;

        while (std::getline(ss, cell, ',')) {
            ++column_count;
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
            if (column_count == 10) {  // Exactly 10 columns expected
                data.push_back(row);
            } else {
                std::cerr << "Warning: Skipping line " << line_num
                          << " due to unexpected number of columns: "
                          << column_count << std::endl;
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
        // Features: first 8 columns plus ocean_proximity (column 9)
        features.push_back({
            row[0], // longitude
            row[1], // latitude
            row[2], // housing_median_age
            row[3], // total_rooms
            row[4], // total_bedrooms
            row[5], // population
            row[6], // households
            row[7], // median_income
            row[9]  // ocean_proximity (numeric)
        });

        // Target: median_house_value (column 8)
        targets.push_back({ row[8] });
    }
}


