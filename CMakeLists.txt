cmake_minimum_required(VERSION 3.28)
project(Perception2D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
set(ORT_LIB_PATH "${ORT_ROOT}/lib/libonnxruntime.so")

find_package(OpenCV REQUIRED)

include_directories(${ORT_ROOT}/include)

add_executable(perception2d_app src_cpp/main.cpp src_cpp/detector.cpp)

target_include_directories(perception2d_app PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(perception2d_app PRIVATE ${OpenCV_LIBS} ${ORT_LIB_PATH})
set_target_properties(perception2d_app PROPERTIES
    BUILD_RPATH "${ORT_ROOT}/lib"
    INSTALL_RPATH "${ORT_ROOT}/lib"
)