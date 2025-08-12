/*
 * tensor.cpp - tensor implementation with automatic differentiation
 * 
 * implements the core tensor functionality including:
 * - memory management and shape calculations
 * - gradient buffer allocation and zeroing
 * - computational graph construction through operator overloading
 * - backpropagation initiation and gradient chain maintenance
 * 
 * IMPORTANT:
 * - gradient buffers allocated lazily to save memory, don't manually allocate them
 * - all operations create new tensors (immutable design)
 * - global graph manager prevents premature tensor destruction
 */

#include "tensor.hpp"
#include "op.hpp"
#include <iostream>
#include "ops/sub.hpp"
#include "ops/mul.hpp"
#include "ops/mean.hpp"
#include "ops/add.hpp"
#include "ops/pow.hpp"
#include "ops/div.hpp"
#include "ops/matmul.hpp"
#include "tensor_ops.hpp"

// global graph manager to keep all tensors and operations alive during computation
// critical for preventing premature destruction of intermediate computation results
extern Graph global_graph;

Tensor::Tensor(std::vector<int> shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_) {
    int total = numel();
    data.resize(total, 0.0f);
    // gradient buffer allocated on-demand when backward() is called to save memory
}

int Tensor::numel() const {
    int n = 1;
    for (int d : shape) n *= d;
    return n;
}

void Tensor::zero_grad() {
    if (requires_grad) {
        if (grad.empty()) {
            // first time calling zero_grad - allocate gradient buffer
            grad.resize(data.size(), 0.0f);
            std::cout << "[Tensor] zero_grad: initialized grad buffer, size=" << grad.size() << std::endl;
        } else {
            // reuse existing buffer - just zero all values
            std::fill(grad.begin(), grad.end(), 0.0f);
            std::cout << "[Tensor] zero_grad: zeroed existing grad buffer, size=" << grad.size() << std::endl;
        }
    }
}

void Tensor::backward() {
    if (!requires_grad) {
        std::cerr << "[Tensor] ERROR: Tensor does not require grad. Backward aborted." << std::endl;
        throw std::runtime_error("Cannot call backward on tensor without requires_grad.");
    }

    if (grad.empty()) {
        // initialize gradient buffer with default gradient of 1.0 (for loss tensor)
        grad.resize(data.size(), 1.0f);
        std::cout << "[Tensor] Initialized grad buffer with size " << grad.size() << std::endl;
    }

    if (auto creator_shared = creator.lock()) {
        // propagate gradients backward through the computation graph
        std::cout << "[Tensor] Calling backward on creator" << std::endl;
        creator_shared->backward(*this);
    } else {
        // no creator means this is a leaf tensor (input or parameter)
        std::cout << "[Tensor] No creator found, this is a leaf tensor" << std::endl;
    }
}

void Tensor::print_data() const {
    std::cout << "Tensor(shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i + 1 != shape.size()) std::cout << ", ";
    }
    std::cout << "], data=[";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i];
        if (i + 1 != data.size()) std::cout << ", ";
    }
    std::cout << "])\n";
}

std::shared_ptr<Tensor> Tensor::detach() const {
    // create tensor copy without gradient tracking for inference
    auto new_tensor = std::make_shared<Tensor>(shape, false);
    new_tensor->data = data;
    return new_tensor;
}

std::shared_ptr<Tensor> Tensor::operator*(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Tensor::operator* shape mismatch");
    }

    auto result = std::make_shared<Tensor>(shape, requires_grad || other.requires_grad);
    result->data.resize(data.size());

    // element-wise multiplication
    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] * other.data[i];

    if (result->requires_grad) {
        // create multiplication operation and link to computational graph
        auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
        auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());
        auto mul_op = std::make_shared<MulOp>(lhs, rhs);
        result->set_creator(mul_op);

        // register with global graph to prevent premature destruction
        global_graph.add_tensor(result);
        global_graph.add_op(mul_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::operator-(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Tensor::operator- shape mismatch");
    }

    auto result = std::make_shared<Tensor>(shape, requires_grad || other.requires_grad);
    result->data.resize(data.size());

    // element-wise subtraction
    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] - other.data[i];

    if (result->requires_grad) {
        // create subtraction operation and link to computational graph
        auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
        auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());
        auto sub_op = std::make_shared<SubOp>(lhs, rhs);
        result->set_creator(sub_op);

        // register with global graph to prevent premature destruction
        global_graph.add_tensor(result);
        global_graph.add_op(sub_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::operator+(const Tensor& other) const {
    if (shape != other.shape) {
        throw std::runtime_error("Tensor::operator+ shape mismatch");
    }

    auto result = std::make_shared<Tensor>(shape, requires_grad || other.requires_grad);
    result->data.resize(data.size());

    // element-wise addition
    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] + other.data[i];

    if (result->requires_grad) {
        // create addition operation and link to computational graph
        auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
        auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());
        auto add_op = std::make_shared<AddOp>(lhs, rhs);
        result->set_creator(add_op);

        // register with global graph to prevent premature destruction
        global_graph.add_tensor(result);
        global_graph.add_op(add_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::pow(float exponent) const {
    auto result = std::make_shared<Tensor>(shape, requires_grad);
    result->data.resize(data.size());

    // element-wise power operation
    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = std::pow(data[i], exponent);

    if (requires_grad) {
        // create power operation and link to computational graph
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        auto pow_op = std::make_shared<PowOp>(self, exponent);
        result->set_creator(pow_op);

        // register with global graph to prevent premature destruction
        global_graph.add_tensor(result);
        global_graph.add_op(pow_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) throw std::runtime_error("Tensor::operator/ division by zero");

    auto result = std::make_shared<Tensor>(shape, requires_grad);
    result->data.resize(data.size());

    // element-wise scalar division
    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] / scalar;

    if (requires_grad) {
        // create division operation and link to computational graph
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        auto div_op = std::make_shared<DivOp>(self, scalar);
        result->set_creator(div_op);

        // register with global graph to prevent premature destruction
        global_graph.add_tensor(result);
        global_graph.add_op(div_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::matmul(const Tensor& other) const {
    auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
    auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());

    // delegate to global matmul function for proper operation creation
    std::shared_ptr<Tensor> result_ptr = ::matmul(lhs, rhs);
    return result_ptr;
}

std::shared_ptr<Tensor> Tensor::mean() const {
    float sum = 0.0f;
    for (float v : data) sum += v;

    auto result = std::make_shared<Tensor>(std::vector<int>{1}, requires_grad);
    result->data[0] = sum / data.size();

    if (requires_grad) {
        // create mean operation and link to computational graph
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());

        auto mean_op = std::make_shared<MeanOp>(self, data.size());
        result->set_creator(mean_op);

        // register with global graph to prevent premature destruction
        global_graph.add_tensor(result);
        global_graph.add_op(mean_op);
    }

    return result;
}

void Tensor::set_creator(std::shared_ptr<Op> op) {
    creator = op;
}
