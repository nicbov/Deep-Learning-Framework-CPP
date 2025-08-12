cmake_minimum_required(VERSION 3.14)
project(cppgrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(cppgrad
    main.cpp
    tensor.cpp
    ops/add.cpp
    ops/matmul.cpp
    ops/mse.cpp
    linear.cpp
    relu.cpp
    src/module.cpp 
    model/sequential.cpp
    optimizer/adam.cpp 
    data/csv_loader.cpp
    ops/linear_op.cpp
    ops/mul.hpp
    ops/sub.hpp
    ops/mean.hpp
    src/module.hpp
    ops/div.cpp
    ops/pow.cpp
    tensor_ops.hpp
    graph.hpp
)
