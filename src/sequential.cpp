#include "sequential.hpp"

void Sequential::add_module(std::shared_ptr<Module> module) {
    modules.push_back(module);
}

std::shared_ptr<Tensor> Sequential::forward(std::shared_ptr<Tensor> input) {
    std::shared_ptr<Tensor> x = input;
    for (size_t i = 0; i < modules.size(); ++i) {
        std::cout << "Forward pass layer " << i << " (" << modules[i]->name() << ")" << std::endl;
        x = modules[i]->forward(x);
    }
    return x;
}

std::vector<std::shared_ptr<Tensor>> Sequential::parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto& module : modules) {
        auto p = module->parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
