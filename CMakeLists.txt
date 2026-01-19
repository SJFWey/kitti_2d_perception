cmake_minimum_required(VERSION 3.20)
project(Perception2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_ROOT "" CACHE PATH "Path to ONNX Runtime root (contains include/ and lib/)")
set(CLI11_ROOT "" CACHE PATH "Path to CLI11 root")

if (NOT ORT_ROOT AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
    set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
endif()
if (NOT CLI11_ROOT AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/cli11")
    set(CLI11_ROOT "${CMAKE_SOURCE_DIR}/third_party/cli11")
endif()

find_package(OpenCV REQUIRED)

find_path(ORT_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ORT_ROOT}" "${ORT_ROOT}/include"
)
find_library(ORT_LIBRARY
    NAMES onnxruntime
    PATHS "${ORT_ROOT}" "${ORT_ROOT}/lib"
)

if (NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Set -DORT_ROOT=/path/to/onnxruntime (must contain include/ and lib/). See README.md for details.")
endif()

find_path(CLI11_INCLUDE_DIR
    NAMES CLI/CLI.hpp CLI11.hpp
    PATHS "${CLI11_ROOT}" "${CLI11_ROOT}/include"
)
if (NOT CLI11_INCLUDE_DIR)
    message(FATAL_ERROR "CLI11 not found. Set -DCLI11_ROOT=/path/to/cli11 (header-only, from https://github.com/CLIUtils/CLI11). See README.md for details.")
endif()


add_executable(perception2d_app
    src_cpp/main.cpp
    src_cpp/core/detector.cpp
    src_cpp/core/sort_tracker.cpp
    src_cpp/core/hungarian.h
    src_cpp/io/input_resolver.cpp
    src_cpp/io/output_writer.cpp
    src_cpp/config/app_config.cpp
    src_cpp/config/ini_config.h
    src_cpp/utils/visualization.cpp)

target_include_directories(perception2d_app PRIVATE
    ${CMAKE_SOURCE_DIR}/src_cpp
    ${OpenCV_INCLUDE_DIRS}
    ${ORT_INCLUDE_DIR}
    ${CLI11_INCLUDE_DIR}
)

target_link_libraries(perception2d_app PRIVATE ${OpenCV_LIBS} ${ORT_LIBRARY})

get_filename_component(ORT_LIB_DIR "${ORT_LIBRARY}" DIRECTORY)
set_target_properties(perception2d_app PROPERTIES
    BUILD_RPATH "${ORT_LIB_DIR}"
    INSTALL_RPATH "${ORT_LIB_DIR}"
)
