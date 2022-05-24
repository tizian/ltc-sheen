cmake_minimum_required (VERSION 3.13...3.18)

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11")
    message(FATAL_ERROR "pybind11 dependency is missing! You probably did not
    clone the project with --recursive. It is possible to recover by calling
    \"git submodule update --init --recursive\"")
endif()

project(sheen)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

set(CMAKE_CXX_STANDARD 17)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ext/pcg32
)

pybind11_add_module(ltcsheen
    THIN_LTO OPT_SIZE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/python_bindings.cpp
)
