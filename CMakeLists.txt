cmake_minimum_required(VERSION 3.14)
project(cppgrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

add_executable(cppgrad
    main.cpp
    src/tensor.cpp
    src/ops/add.cpp
    src/ops/mul.cpp 
    src/ops/matmul.cpp
    src/ops/mse.cpp
    src/ops/relu.cpp
    src/linear.cpp
    src/optimizer.cpp
)
