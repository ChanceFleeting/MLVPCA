cmake_minimum_required(VERSION 3.12)

project(fun_GetCov_SeqADMM_SelectTuningPar)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find Python and pybind11
find_package(Python3 3.6 REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)

# Find Armadillo
find_package(Armadillo REQUIRED)

# Add the module
add_library(fun_GetCov_SeqADMM_SelectTuningPar MODULE module.cpp)

# Include and link libraries
target_include_directories(fun_GetCov_SeqADMM_SelectTuningPar PRIVATE ${Python3_INCLUDE_DIRS} ${pybind11_INCLUDE_DIRS} ${Armadillo_INCLUDE_DIRS})
target_link_libraries(fun_GetCov_SeqADMM_SelectTuningPar PRIVATE Python3::Python pybind11::module Armadillo::Armadillo)

# Set output properties
set_target_properties(fun_GetCov_SeqADMM_SelectTuningPar PROPERTIES PREFIX "" SUFFIX ".so")

# Add warning flags (optional)
target_compile_options(fun_GetCov_SeqADMM_SelectTuningPar PRIVATE -Wall -Wextra -Wpedantic)

# Usage instructions
# mkdir build
# cd build
# cmake ..
# make
