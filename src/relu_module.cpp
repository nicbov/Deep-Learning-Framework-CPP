#include "relu_module.hpp"
#include "relu.hpp" 

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    return relu(input);  
}
