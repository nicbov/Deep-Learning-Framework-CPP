#include <iostream>
#include <iomanip>
#include <vector> // Added for std::vector

int main() {
    std::cout << "=== Testing Normalization Issues ===" << std::endl;
    
    // Your actual normalization parameters
    const float feature_mins[9] = {
        -124.35f, 32.54f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 0.4999f, -1.0f
    };
    const float feature_maxs[9] = {
        -114.31f, 41.95f, 52.0f, 39320.0f, 6445.0f, 35682.0f, 6082.0f, 15.0001f, 2.0f
    };
    
    std::cout << "\nFeature ranges:" << std::endl;
    for (int i = 0; i < 9; ++i) {
        float range = feature_maxs[i] - feature_mins[i];
        std::cout << "Feature " << i << ": range = " << range << std::endl;
    }
    
    // Test with some actual data values
    std::cout << "\nTesting with actual data values:" << std::endl;
    
    // Sample some actual values from your data
    std::vector<float> raw_values = {
        -122.23f,  // longitude
        37.88f,    // latitude  
        41.0f,     // housing_median_age
        880.0f,    // total_rooms
        129.0f,    // total_bedrooms
        322.0f,    // population
        126.0f,    // households
        8.3252f,   // median_income
        1.0f       // ocean_proximity
    };
    
    std::vector<float> normalized_values(9);
    
    for (int i = 0; i < 9; ++i) {
        float raw_val = raw_values[i];
        float norm_val = (raw_val - feature_mins[i]) / (feature_maxs[i] - feature_mins[i]);
        normalized_values[i] = norm_val;
        
        std::cout << "Feature " << i << ": raw=" << raw_val 
                  << " -> normalized=" << std::fixed << std::setprecision(6) << norm_val << std::endl;
    }
    
    // Now test what happens when we multiply by weights
    std::cout << "\nTesting weight multiplication:" << std::endl;
    
    // Simulate some weight values (similar to what your Linear layer generates)
    std::vector<float> weights = {-0.25f, 0.59f, 0.90f, -0.63f, 0.46f, 0.56f, 0.20f, 0.19f, -0.69f};
    
    float result = 0.0f;
    for (int i = 0; i < 9; ++i) {
        float contribution = normalized_values[i] * weights[i];
        result += contribution;
        std::cout << "Feature " << i << " contribution: " << normalized_values[i] << " * " << weights[i] 
                  << " = " << contribution << std::endl;
    }
    
    std::cout << "\nTotal result: " << result << std::endl;
    
    // Test what happens with extreme values
    std::cout << "\nTesting with extreme values:" << std::endl;
    
    // What if we have a house with 30000 rooms?
    float extreme_rooms = 30000.0f;
    float extreme_norm = (extreme_rooms - feature_mins[3]) / (feature_maxs[3] - feature_mins[3]);
    std::cout << "Extreme rooms: " << extreme_rooms << " -> normalized: " << extreme_norm << std::endl;
    
    // This normalized value is very close to 1.0, which means it's at the extreme end
    // When multiplied by weights, it can still produce large values
    
    return 0;
}
