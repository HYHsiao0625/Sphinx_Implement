cmake_minimum_required(VERSION 3.12)

project(cnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Uncomment this section if you plan to use CUDA
# set(CMAKE_CUDA_STANDARD 11)
# set(CMAKE_CUDA_STANDARD_REQUIRED True)

# CUDA Architecture Setup
# if(NOT DEFINED CMAKE_CUDA_ARCH)
#    message(FATAL_ERROR "CMAKE_CUDA_ARCH is not set. Use: -DCMAKE_CUDA_ARCH=[KEPLER|MAXWELL|PASCAL|VOLTA|TURING|AMPERE].")
# endif()

# Uncomment the CUDA architecture block if you need to configure it
# set(CMAKE_CUDA_ARCHITECTURES 60 61)  # Default to Pascal, adjust as needed

# Executables
add_executable(cnn src/cnn.cpp src/ckks.cpp)
add_executable(keygen src/keygen.cpp src/ckks.cpp)

# Include directories
target_include_directories(cnn PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_include_directories(keygen PUBLIC ${CMAKE_SOURCE_DIR}/src)

# SEAL library
find_package(SEAL REQUIRED)

# OpenMP support
find_package(OpenMP REQUIRED)

# Combine link libraries for both executables
if(OpenMP_CXX_FOUND)
    target_compile_options(cnn PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(cnn PRIVATE SEAL::seal ${OpenMP_CXX_LIBRARIES})

    target_compile_options(keygen PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(keygen PRIVATE SEAL::seal ${OpenMP_CXX_LIBRARIES})
else()
    target_link_libraries(cnn PRIVATE SEAL::seal)
    target_link_libraries(keygen PRIVATE SEAL::seal)
endif()

# Uncomment the properties for CUDA if needed
# set_target_properties(cnn PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_STANDARD 11
# )
