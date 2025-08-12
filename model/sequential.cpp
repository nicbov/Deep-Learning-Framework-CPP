/*
 * sequential.cpp - sequential model implementation
 * 
 * implements the forward pass and parameter collection for sequential models:
 * - forward pass applies modules in sequence: input -> module1 -> module2 -> ... -> output
 * - parameter collection aggregates parameters from all contained modules
 * - provides debugging output during forward pass execution
 * each module's output becomes the next module's input, creating a chain of transformations
 */

#include "sequential.hpp"

void Sequential::add_module(std::shared_ptr<Module> module) {
    // add module to the end of the sequential chain
    modules.push_back(module);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    // thjs executes modules sequentially: input -> module1 -> module2 -> ... -> output
    std::shared_ptr<Tensor> x = input;
    for (size_t i = 0; i < modules.size(); ++i) {
        std::cout << "Forward pass layer " << i << " (" << modules[i]->name() << ")" << std::endl;
        x = modules[i]->forward(x);
    }
    return x;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() const {
    // thsi collects parameters from all contained modules
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& module : modules) {
        auto p = module->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
