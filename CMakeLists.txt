cmake_minimum_required(VERSION 3.12)

# project(cnn)
project(cnn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT DEFINED CMAKE_CUDA_ARCH)
    message(FATAL_ERROR "CMAKE_CUDA_ARCH is not set. Use: -DCMAKE_CUDA_ARCH=[KEPLER|MAXWELL|PASCAL|VOLTA|TURING|AMPERE].")
endif()

if(CMAKE_CUDA_ARCH STREQUAL "PASCAL")
    set(CMAKE_CUDA_ARCHITECTURES 60 61)     # Pascal GPUs: GTX 10XX series
elseif(CMAKE_CUDA_ARCH STREQUAL "MAXWELL")
    set(CMAKE_CUDA_ARCHITECTURES 50 52 53)  # Maxwell GPUs: GTX 9XX series
elseif(CMAKE_CUDA_ARCH STREQUAL "KEPLER")
    set(CMAKE_CUDA_ARCHITECTURES 35 37)     # Kepler GPUs: GTX 6XX, GTX 7XX, GTX Titan
elseif(CMAKE_CUDA_ARCH STREQUAL "VOLTA")
    set(CMAKE_CUDA_ARCHITECTURES 70)        # Volta GPUs: Tesla V100, GTX Titan V
elseif(CMAKE_CUDA_ARCH STREQUAL "TURING")
    set(CMAKE_CUDA_ARCHITECTURES 75 86)     # Turing GPUs: GTX 16XX, RTX 20XX
elseif(CMAKE_CUDA_ARCH STREQUAL "AMPERE")
    set(CMAKE_CUDA_ARCHITECTURES 80 86)     # Ampere GPUs: RTX 30XX series
else()
    message(FATAL_ERROR "Unknown CMAKE_CUDA_ARCH: ${CMAKE_CUDA_ARCH}.")
endif()

# add_executable(cnn src/cnn.cpp src/ckks.cpp)
# add_executable(keygen src/keygen.cpp src/ckks.cpp)
add_executable(cnn src/cnn.cpp src/ckks.cpp)
add_executable(keygen src/keygen.cpp src/ckks.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)

find_package(SEAL)
target_link_libraries(${PROJECT_NAME} SEAL::seal)
target_link_libraries(keygen SEAL::seal)

set_target_properties(cnn PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 11
)