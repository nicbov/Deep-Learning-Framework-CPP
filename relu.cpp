/*
 * relu.cpp - rectified linear unit activation function implementation
 * 
 * this file implements the relu activation function which is critical for neural network training:
 * - forward pass: applies f(x) = max(0, x) element-wise to prevent negative activations
 * - backward pass: gradient is 1 if input > 0, 0 otherwise (simple but effective)
 * - integrates with global computation graph for automatic differentiation
 * 
 * IMPORTANT insight: relu is computationally efficient and helps prevent vanishing gradients
 * the gradient computation is basically trivial but the activation pattern is crucial for deep networks
 */

#include "relu.hpp"
#include "tensor.hpp"
#include "graph.hpp"
#include "op.hpp"

// global graph manager prevents premature destruction of tensors during computation
// critical for maintaining computational graph integrity across forward/backward passes
extern Graph global_graph;

// constructor stores input tensor reference for gradient computation during backpropagation
// weak references prevent circular dependencies while maintaining access to input data
ReLUOp::ReLUOp(std::shared_ptr<Tensor> input) : input(input) {}

std::shared_ptr<Tensor> ReLUOp::forward() {
    // create output tensor with same shape and gradient requirements as input
    // this maintains the computational graph structure for backpropagation
    output = std::make_shared<Tensor>(input->shape, input->requires_grad);
    
    // apply relu activation element-wise: max(0, x)
    // this creates the "dead relu" problem where negative inputs produce zero gradients
    for (size_t i = 0; i < input->data.size(); ++i) {
        output->data[i] = std::max(0.0f, input->data[i]);
    }
    
    return output;
}

void ReLUOp::backward(Tensor& grad_output) {
    if (!input->requires_grad) return;
    
    // initialize gradient buffer if needed (lazy allocation to save memory)
    // gradient buffers are only allocated when actually needed for backpropagation
    if (input->grad.empty()) {
        input->grad.resize(input->data.size(), 0.0f);
    }
    
    // compute gradients w.r.t. input: ∂relu/∂x = 1 if x > 0, else 0
    // this is the key insight: relu gradient is discontinuous at x=0 but rarely causes issues
    for (size_t i = 0; i < input->data.size(); ++i) {
        // relu gradient: 1 if input > 0, 0 otherwise
        // this creates sparse gradients which can help with feature selection
        float grad = (input->data[i] > 0.0f) ? grad_output.grad[i] : 0.0f;
        input->grad[i] += grad;
    }

    // propagate gradients to input tensor's creator to maintain gradient chain
    // critical fix: check if creator is different to prevent infinite loops
    auto input_creator = input->creator.lock();
    if (input_creator && input_creator.get() != this) {
        // only propagate if the creator is different from this operation
        input_creator->backward(*input);
    }
}

std::shared_ptr<Tensor> ReLU::forward(std::shared_ptr<Tensor> input) {
    // create a proper relu operation to maintain gradient chain
    // this separates the module interface from the operation implementation
    auto op = std::make_shared<ReLUOp>(input);
    auto result = op->forward();
    
    // set the creator after forward to ensure proper gradient chain
    // this links the output tensor to its creating operation for backpropagation
    if (input->requires_grad) {
        result->set_creator(op);
    }
    
    // register tensor and operation with global graph for lifetime management
    // prevents premature destruction of intermediate computation results
    global_graph.add_tensor(result);
    global_graph.add_op(op);
    
    return result;
}
