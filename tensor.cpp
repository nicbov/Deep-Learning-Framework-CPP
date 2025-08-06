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

//my global graph manager to keep everything alive
extern Graph global_graph;

Tensor::Tensor(std::vector<int> shape_, bool requires_grad_)
    : shape(shape_), requires_grad(requires_grad_) {
    int total = numel();
    data.resize(total, 0.0f);
    if (requires_grad) {
        // Don't pre-allocate grad buffer - let backward() handle it
        std::cout << "[Tensor ctor] requires_grad=true, grad buffer will be allocated on demand" << std::endl;
    } else {
        std::cout << "[Tensor ctor] requires_grad=false" << std::endl;
    }
}

int Tensor::numel() const {
    int n = 1;
    for (int d : shape) n *= d;
    return n;
}

void Tensor::zero_grad() {
    if (requires_grad) {
        std::fill(grad.begin(), grad.end(), 0.0f);
        std::cout << "[Tensor] zero_grad called, grad zeroed" << std::endl;
    }
}

void Tensor::backward() {
    std::cout << "\n========== BACKWARD ==========\n";
    std::cout << "[Tensor] backward() called on Tensor with shape: ";
    for (auto d : shape) std::cout << d << " ";
    std::cout << "\n";
    std::cout << "[Tensor::backward] Tensor address: " << this << "\n";

    if (!requires_grad) {
        std::cerr << "[Tensor] ERROR: Tensor does not require grad. Backward aborted.\n";
        throw std::runtime_error("Cannot call backward on tensor without requires_grad.");
    }

    if (creator.expired()) {
        std::cout << "[Tensor] No creator set (nullptr). Assuming this is a leaf node.\n";
    } else {
        std::cout << "[Tensor] creator is valid\n";
    }

    if (grad.empty()) {
        grad.resize(data.size(), 1.0f);
        std::cout << "[Tensor] Gradient buffer initialized to 1.0 (likely loss tensor)\n";
    } else {
        std::cout << "[Tensor] Gradient already initialized, size = " << grad.size() << "\n";
    }

    if (auto creator_shared = creator.lock()) {
        std::cout << "[Tensor] Creator is valid. Calling Op::backward on Op at address: " << creator_shared.get() << "\n";
        creator_shared->backward(*this);
    } else {
        std::cout << "[Tensor] No valid creator found, stopping here.\n";
    }

    std::cout << "========== END BACKWARD ==========\n\n";
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
    std::cout << "[Tensor] detach() called, requires_grad set to false, creator removed" << std::endl;
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

    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] * other.data[i];

    if (result->requires_grad) {
        auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
        auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());
        auto mul_op = std::make_shared<MulOp>(lhs, rhs);
        result->set_creator(mul_op);

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

    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] - other.data[i];

    if (result->requires_grad) {
        auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
        auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());
        auto sub_op = std::make_shared<SubOp>(lhs, rhs);
        result->set_creator(sub_op);

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

    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] + other.data[i];

    if (result->requires_grad) {
        auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
        auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());
        auto add_op = std::make_shared<AddOp>(lhs, rhs);
        result->set_creator(add_op);

        global_graph.add_tensor(result);
        global_graph.add_op(add_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::pow(float exponent) const {
    auto result = std::make_shared<Tensor>(shape, requires_grad);
    result->data.resize(data.size());

    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = std::pow(data[i], exponent);

    if (result->requires_grad) {
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        auto pow_op = std::make_shared<PowOp>(self, exponent);
        result->set_creator(pow_op);

        global_graph.add_tensor(result);
        global_graph.add_op(pow_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::operator/(float scalar) const {
    if (scalar == 0.0f) throw std::runtime_error("Tensor::operator/ division by zero");

    auto result = std::make_shared<Tensor>(shape, requires_grad);
    result->data.resize(data.size());

    for (size_t i = 0; i < data.size(); ++i)
        result->data[i] = data[i] / scalar;

    if (result->requires_grad) {
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());
        auto div_op = std::make_shared<DivOp>(self, scalar);
        result->set_creator(div_op);

        global_graph.add_tensor(result);
        global_graph.add_op(div_op);
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::matmul(const Tensor& other) const {
    auto lhs = std::const_pointer_cast<Tensor>(shared_from_this());
    auto rhs = std::const_pointer_cast<Tensor>(other.shared_from_this());

    std::shared_ptr<Tensor> result_ptr = ::matmul(lhs, rhs);
    return result_ptr;
}

std::shared_ptr<Tensor> Tensor::mean() const {
    float sum = 0.0f;
    for (float v : data) sum += v;

    auto result = std::make_shared<Tensor>(std::vector<int>{1}, requires_grad);
    result->data[0] = sum / data.size();

    if (requires_grad) {
        auto self = std::const_pointer_cast<Tensor>(shared_from_this());

        auto mean_op = std::make_shared<MeanOp>(self, data.size());
        result->set_creator(mean_op);

        std::cout << "[mean()] result tensor: " << result.get() << "\n";
        if (result->creator.expired()) {
            std::cout << "[mean()] WARNING: creator expired\n";
        } else {
            std::cout << "[mean()] creator set successfully\n";
        }

        //holding mean_op alive to prevent premature destruction
        static std::vector<std::shared_ptr<Op>> op_hold;
        op_hold.push_back(mean_op);

        global_graph.add_tensor(result);
        global_graph.add_op(mean_op);
    }

    return result;
}

void Tensor::set_creator(std::shared_ptr<Op> op) {
    std::cout << "[Tensor] set_creator called with op: " << typeid(*op.get()).name() << "\n";
    creator = op;
}
