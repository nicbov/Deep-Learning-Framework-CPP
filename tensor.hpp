#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include <typeinfo>

class Op;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<int> shape;
    std::vector<float> data;

    bool requires_grad = false;
    std::vector<float> grad;

    std::weak_ptr<Op> creator;

    // Tensor operations
    std::shared_ptr<Tensor> operator-(const Tensor& other) const;
    std::shared_ptr<Tensor> operator*(const Tensor& other) const;
    std::shared_ptr<Tensor> operator+(const Tensor& other) const;
    std::shared_ptr<Tensor> pow(float exponent) const;
    std::shared_ptr<Tensor> operator/(float scalar) const;
    std::shared_ptr<Tensor> matmul(const Tensor& other) const;
    std::shared_ptr<Tensor> mean() const;

    // Construction
    Tensor(std::vector<int> shape, bool requires_grad = false);

    // Grad support
    int numel() const;
    void zero_grad();
    void backward();
    void print_data() const;
    std::shared_ptr<Tensor> detach() const;

    // Backprop graph linkage
    void set_creator(std::shared_ptr<Op> op);
};
