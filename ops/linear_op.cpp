#include "linear_op.hpp"
#include <iostream>

LinearOp::LinearOp(const std::shared_ptr<Tensor>& input_,
    const std::shared_ptr<Tensor>& weight_,
    const std::shared_ptr<Tensor>& bias_) :
input(input_), weight(weight_), bias(bias_) {
inputs.push_back(input);
inputs.push_back(weight);
if (bias) inputs.push_back(bias);
}

void LinearOp::backward(Tensor& grad_output) {
    std::cout << "[LinearOp] backward called\n";

    int batch = input->shape[0];
    int in_dim = input->shape[1];
    int out_dim = weight->shape[1];

    // Check grad_output size matches batch*out_dim
    if (grad_output.grad.size() != batch * out_dim) {
        std::cerr << "[LinearOp] ERROR: grad_output.grad size mismatch\n";
        return;
    }

    if (input->requires_grad) {
        if (input->grad.size() != input->data.size()) input->grad.assign(input->data.size(), 0.0f);
        else std::fill(input->grad.begin(), input->grad.end(), 0.0f);

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < in_dim; ++i) {
                float grad_val = 0.0f;
                for (int j = 0; j < out_dim; ++j) {
                    grad_val += grad_output.grad[b * out_dim + j] * weight->data[i * out_dim + j];
                }
                input->grad[b * in_dim + i] = grad_val;  // No +=, overwrite after zeroing
            }
        }

        std::cout << "[LinearOp] input grad sample: ";
        for (int i = 0; i < std::min(5, (int)input->grad.size()); ++i) {
            std::cout << input->grad[i] << " ";
        }
        std::cout << std::endl;
    }

    if (weight->requires_grad) {
        if (weight->grad.size() != weight->data.size()) weight->grad.assign(weight->data.size(), 0.0f);
        else std::fill(weight->grad.begin(), weight->grad.end(), 0.0f);

        for (int i = 0; i < in_dim; ++i) {
            for (int j = 0; j < out_dim; ++j) {
                float grad_val = 0.0f;
                for (int b = 0; b < batch; ++b) {
                    grad_val += input->data[b * in_dim + i] * grad_output.grad[b * out_dim + j];
                }
                weight->grad[i * out_dim + j] = grad_val; // overwrite after zeroing
            }
        }

        std::cout << "[LinearOp] weight grad sample: ";
        for (int i = 0; i < std::min(5, (int)weight->grad.size()); ++i) {
            std::cout << weight->grad[i] << " ";
        }
        std::cout << std::endl;
    }

    if (bias && bias->requires_grad) {
        if (bias->grad.size() != bias->data.size()) bias->grad.assign(bias->data.size(), 0.0f);
        else std::fill(bias->grad.begin(), bias->grad.end(), 0.0f);

        for (int j = 0; j < out_dim; ++j) {
            float grad_val = 0.0f;
            for (int b = 0; b < batch; ++b) {
                grad_val += grad_output.grad[b * out_dim + j];
            }
            bias->grad[j] = grad_val; // overwrite after zeroing
        }

        std::cout << "[LinearOp] bias grad sample: ";
        for (int i = 0; i < std::min(5, (int)bias->grad.size()); ++i) {
            std::cout << bias->grad[i] << " ";
        }
        std::cout << std::endl;
    }

    if (auto creator = input->creator.lock()) {
        std::cout << "[LinearOp] calling input creator backward\n";
        creator->backward(*input);
    } else {
        std::cout << "[LinearOp] input creator not found or expired, backward ends here\n";
    }
}
