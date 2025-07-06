#pragma once

#include <vector>
#include <memory>
#include <iostream>

class Op;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    std::vector<int> shape;
    std::vector<float> data;

    bool requires_grad = false;
    std::vector<float> grad;

    std::shared_ptr<Op> creator = nullptr;

    Tensor(std::vector<int> shape, bool requires_grad = false);

    int numel() const;
    void zero_grad();
    void backward();
    void print_data() const;
};
