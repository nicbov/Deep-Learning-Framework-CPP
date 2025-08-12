# CPPGrad: A C++ Neural Network Framework with Automatic Differentiation

A modern, educational C++ implementation of a neural network framework featuring automatic differentiation, computational graphs, and a complete training pipeline. Built from scratch to demonstrate deep learning fundamentals while maintaining production-quality code structure.

## Features

- **Automatic Differentiation**: Full backward pass implementation with computational graph tracking
- **Tensor Operations**: Multi-dimensional arrays with gradient computation support
- **Neural Network Layers**: Linear layers, ReLU activations, and sequential model containers
- **Optimization**: Adam optimizer with momentum and adaptive learning rates
- **Data Pipeline**: CSV loading, preprocessing, and normalization utilities
- **Memory Management**: Intelligent computational graph lifecycle management
- **Training Loop**: Complete training pipeline with early stopping and monitoring

## Architecture

### Core Components

- **`Tensor`**: Multi-dimensional arrays with automatic gradient tracking
- **`Op`**: Base class for all computational operations
- **`Module`**: Abstract interface for neural network layers
- **`Graph`**: Computational graph memory manager
- **`Sequential`**: Container for chaining neural network modules

### Neural Network Operations

- **Matrix Multiplication**: Core linear transformations
- **Element-wise Operations**: Addition, subtraction, multiplication, division
- **Activation Functions**: ReLU implementation
- **Loss Functions**: Mean Squared Error (MSE)
- **Reduction Operations**: Mean computation

### Training Infrastructure

- **Adam Optimizer**: Adaptive moment estimation with momentum
- **Data Loading**: Robust CSV parsing and validation
- **Normalization**: Min-max scaling for stable training
- **Monitoring**: Gradient flow analysis and parameter tracking
- **Early Stopping**: Prevents overfitting with patience-based stopping

## Project Structure

cppgrad/
├── CMakeLists.txt              # Build configuration
├── main.cpp                    # Training script and main entry point
├── tensor.hpp                  # Core tensor class definition
├── tensor.cpp                  # Tensor implementation
├── op.hpp                      # Base operation class
├── graph.hpp                   # Computational graph manager
├── src/
│   ├── module.hpp             # Base neural network module
│   └── module.cpp             # Module implementation
├── model/
│   ├── sequential.hpp         # Sequential model container
│   └── sequential.cpp         # Sequential implementation
├── ops/                       # Neural network operations
│   ├── add.cpp/hpp           # Addition operation
│   ├── sub.cpp/hpp           # Subtraction operation
│   ├── mul.cpp/hpp           # Multiplication operation
│   ├── div.cpp/hpp           # Division operation
│   ├── matmul.cpp/hpp        # Matrix multiplication
│   ├── mse.cpp/hpp           # Mean squared error loss
│   ├── mean.hpp              # Mean reduction
│   ├── pow.cpp/hpp           # Power operation
│   └── linear_op.cpp/hpp     # Linear layer operation
├── optimizer/
│   ├── adam.cpp              # Adam optimizer implementation
│   └── adam.hpp              # Adam optimizer interface
├── data/
│   ├── csv_loader.cpp        # CSV data loading utilities
│   ├── csv_loader.hpp        # Data loading interface
│   └── housing_clean.csv     # California housing dataset
├── linear.cpp                 # Linear layer implementation
├── linear.hpp                 # Linear layer interface
├── relu.cpp                   # ReLU activation implementation
├── relu.hpp                   # ReLU activation interface
└── tensor_ops.hpp             # Tensor operation utilities


## Building and Running

### Prerequisites

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.14** or higher
- **Make** or **Ninja** build system

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/nicbov/Deep-Learning-Framework-CPP.git
cd cppgrad

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)

# Run the training script
./cppgrad
```

### Alternative Build Methods

```bash
# Using Ninja (faster builds)
cmake -G Ninja ..
ninja

# Using specific compiler
cmake -DCMAKE_CXX_COMPILER=g++-9 ..
make
```

## Usage Example

The main training script demonstrates a complete neural network training pipeline:

```cpp
#include "model/sequential.hpp"
#include "optimizer/adam.hpp"
#include "data/csv_loader.hpp"

int main() {
    // Load and preprocess California housing data
    auto data = load_csv("data/housing_clean.csv");
    
    // Build neural network: 9 -> 16 -> 8 -> 4 -> 1
    auto model = std::make_shared<Sequential>();
    model->add_module(std::make_shared<Linear>(9, 16));
    model->add_module(std::make_shared<ReLU>());
    model->add_module(std::make_shared<Linear>(16, 8));
    model->add_module(std::make_shared<ReLU>());
    model->add_module(std::make_shared<Linear>(8, 4));
    model->add_module(std::make_shared<ReLU>());
    model->add_module(std::make_shared<Linear>(4, 1));
    
    // Train with Adam optimizer
    Adam optimizer(0.01f);
    
    // Training loop with early stopping
    for (int epoch = 0; epoch < 200; ++epoch) {
        model->zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss->backward();
        optimizer.step(model->parameters());
    }
    
    return 0;
}
```

## Dataset

The project includes the **California Housing Dataset** (`housing_clean.csv`) with the following features:

- **Geographic**: longitude, latitude
- **Property**: housing_median_age, total_rooms, total_bedrooms
- **Demographic**: population, households, median_income
- **Categorical**: ocean_proximity (encoded as numeric)
- **Target**: median_house_value (regression target)

**Dataset Statistics**:
- **Samples**: 20,434 housing records
- **Features**: 9 input dimensions
- **Target Range**: $14,999 - $500,001
- **Preprocessing**: Cleaned and normalized for training

## 🔬 Technical Details

### Automatic Differentiation

The framework implements reverse-mode automatic differentiation:

1. **Forward Pass**: Creates computational graph with operation nodes
2. **Gradient Computation**: Each operation knows how to compute gradients w.r.t. inputs
3. **Backward Pass**: Propagates gradients through the graph using chain rule
4. **Memory Management**: Graph manager prevents premature tensor destruction

### Neural Network Architecture

- **Input Layer**: 9 features (housing characteristics)
- **Hidden Layer 1**: 16 neurons with ReLU activation
- **Hidden Layer 2**: 8 neurons with ReLU activation  
- **Hidden Layer 3**: 4 neurons with ReLU activation
- **Output Layer**: 1 neuron (house price prediction)

### Training Configuration

- **Optimizer**: Adam with learning rate 0.01
- **Loss Function**: Mean Squared Error (MSE)
- **Normalization**: Min-max scaling to [0,1] range
- **Early Stopping**: 20 epochs patience
- **Gradient Clipping**: Maximum norm 1.0
- **Maximum Epochs**: 200

## Testing and Validation

The project includes comprehensive testing and debugging features:

- **Gradient Monitoring**: Tracks gradient flow through the network
- **Parameter Tracking**: Monitors weight changes during training
- **Prediction Validation**: Shows predictions vs. targets every 10 epochs
- **Memory Leak Prevention**: Computational graph cleanup after each step
- **Numerical Stability**: NaN/Inf detection and early stopping

