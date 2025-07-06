#include "matmul.hpp"

MatMulOp::MatMulOp(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    inputs.push_back(a);
    inputs.push_back(b);
}

void MatMulOp::backward(Tensor& grad_output) {
    auto a = inputs[0];
    auto b = inputs[1];

    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    if (a->requires_grad) {
        if (a->grad.empty()) a->grad.resize(a->data.size(), 0.0f);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                for (int l = 0; l < n; ++l) {
                    a->grad[i * k + j] += grad_output.grad[i * n + l] * b->data[j * n + l];
                }
            }
        }
    }

    if (b->requires_grad) {
        if (b->grad.empty()) b->grad.resize(b->data.size(), 0.0f);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int l = 0; l < m; ++l) {
                    b->grad[i * n + j] += a->data[l * k + i] * grad_output.grad[l * n + j];
                }
            }
        }
    }

    if (a->creator) a->backward();
    if (b->creator) b->backward();
}

std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b) {
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];

    auto result = std::make_shared<Tensor>(std::vector<int>{m, n}, a->requires_grad || b->requires_grad);
    result->data.resize(m * n, 0.0f);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                result->data[i * n + j] += a->data[i * k + l] * b->data[l * n + j];
            }
        }
    }

    if (result->requires_grad) {
        auto op = std::make_shared<MatMulOp>(a, b);
        result->creator = op;
    }

    return result;
}
