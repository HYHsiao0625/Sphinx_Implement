cmake_minimum_required(VERSION 3.12)

project(CNN)

add_executable(CNN src/cnn.cpp)
add_executable(SERVER src/server.cpp)
add_executable(CLIENT src/client.cpp)