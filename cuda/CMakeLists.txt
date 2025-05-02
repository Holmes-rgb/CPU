cmake_minimum_required(VERSION 3.30)
project(cuda CUDA)
project(vector CUDA)
project(matrix CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cuda cuda-hello-world.cu)
add_executable(vector vector-addition.cu)
add_executable(matrix matrix-vector-product.cu)

set_target_properties(cuda PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vector PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
