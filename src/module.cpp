#include "module.hpp"

void Module::zero_grad() {
    for (auto& param : parameters()) {
        param->zero_grad();
    }
}
