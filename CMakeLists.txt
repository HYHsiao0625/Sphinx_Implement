cmake_minimum_required(VERSION 3.12)

project(cnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(cnn src/cnn.cpp src/ckks.cpp)
add_executable(server src/server.cpp src/ckks.cpp)
add_executable(client src/client.cpp src/ckks.cpp)
add_executable(keygen src/keygen.cpp src/ckks.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)

find_package(SEAL)
target_link_libraries(${PROJECT_NAME} SEAL::seal)
target_link_libraries(server SEAL::seal)
target_link_libraries(client SEAL::seal)
target_link_libraries(keygen SEAL::seal)