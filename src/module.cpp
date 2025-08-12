/*
 * module.cpp - base module implementation
 * 
 * implements the common functionality shared by all neural network modules:
 * - gradient zeroing: resets all parameter gradients to zero
 * - parameter iteration: accesses all trainable parameters
 * 
 * IMPORTANT: gradient zeroing is essential for the proper backpropagation
 * without this, gradients would accumulate across multiple forward/backward passes leading to Nan or inf values
 */

#include "module.hpp"

void Module::zero_grad() {
    // iterate through all parameters and zero their gradients
    for (auto& param : parameters()) {
        param->zero_grad();
    }
}
