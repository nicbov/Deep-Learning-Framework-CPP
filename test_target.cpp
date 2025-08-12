#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "=== Testing Target Normalization ===" << std::endl;
    
    // Your target normalization parameters
    const float target_min = 14999.0f;
    const float target_max = 500001.0f;
    
    std::cout << "Target range: " << target_min << " to " << target_max << std::endl;
    std::cout << "Target range size: " << (target_max - target_min) << std::endl;
    
    // Test with some actual target values from your output
    std::vector<float> target_prices = {452600.0f, 358500.0f, 352100.0f, 341300.0f, 342200.0f};
    
    std::cout << "\nTesting target normalization:" << std::endl;
    for (size_t i = 0; i < target_prices.size(); ++i) {
        float raw_price = target_prices[i];
        float norm_price = (raw_price - target_min) / (target_max - target_min);
        
        std::cout << "Target " << i << ": raw=$" << raw_price 
                  << " -> normalized=" << std::fixed << std::setprecision(6) << norm_price << std::endl;
        
        // Check if normalized value is reasonable
        if (norm_price < 0.0f || norm_price > 1.0f) {
            std::cout << "  WARNING: Normalized value outside [0,1] range!" << std::endl;
        }
    }
    
    // Test denormalization
    std::cout << "\nTesting denormalization:" << std::endl;
    std::vector<float> test_norm_values = {0.0f, 0.5f, 1.0f, -0.1f, 1.1f};
    
    for (float norm_val : test_norm_values) {
        float denorm_price = norm_val * (target_max - target_min) + target_min;
        std::cout << "Normalized " << norm_val << " -> denormalized $" << denorm_price << std::endl;
    }
    
    // Test what happens with extreme normalized values
    std::cout << "\nTesting extreme normalized values:" << std::endl;
    std::vector<float> extreme_values = {-10.0f, 10.0f, 100.0f, -100.0f};
    
    for (float extreme_val : extreme_values) {
        float denorm_price = extreme_val * (target_max - target_min) + target_min;
        std::cout << "Extreme normalized " << extreme_val << " -> denormalized $" << denorm_price << std::endl;
    }
    
    return 0;
}
