cmake_minimum_required(VERSION 3.12)

project(CNN)

add_executable(CNN cnn.cpp)
add_executable(SERVER server.cpp)
add_executable(CLIENT client.cpp)
add_executable(OUTPUT main.cpp)