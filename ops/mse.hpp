/*
 * mse.hpp - mean squared error loss function for regression tasks
 * 
 * this loss function is essential for training neural networks on regression problems:
 * - computes the average squared difference between predictions and targets
 * - provides smooth gradients that enable stable training and convergence
 * - used extensively in housing price prediction, function approximation, and time series forecasting
 * - critical for measuring model performance and guiding parameter updates during training
 */

#pragma once

#include "../op.hpp"
#include "../tensor.hpp"

// computes mean squared error loss between predictions and targets
// this function creates a computational graph that enables automatic differentiation
// the loss value is used by the optimizer to update model parameters during training
std::shared_ptr<Tensor> mse_loss(const std::shared_ptr<Tensor>& pred, const std::shared_ptr<Tensor>& target);
