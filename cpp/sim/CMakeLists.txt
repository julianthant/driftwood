cmake_minimum_required(VERSION 3.15...3.30)
project(driftwood_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_sim src/bindings.cpp)

if(MSVC)
  target_compile_options(_sim PRIVATE /W4)
else()
  target_compile_options(_sim PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS _sim DESTINATION driftwood)
