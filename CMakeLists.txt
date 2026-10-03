cmake_minimum_required(VERSION 4.3)
project(DigitsGAN)

set(CMAKE_CXX_STANDARD 26)

add_executable(DigitsGAN main.cpp
        math_vector.hpp
        matrix.hpp
        neural_network.hpp
        neural_network.cpp
        conditional_gan.hpp
        conditional_gan.cpp)
