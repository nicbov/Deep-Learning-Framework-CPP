#pragma once

#include "../op.hpp"
#include "../tensor.hpp"

std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target);
