cmake_minimum_required(VERSION 3.18)
project(gpu_canny LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# stb_image (header-only, implementation in stb_image_impl.cpp).
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# CUDA object library: fused kernel (Kushaan's territory).
add_library(canny_cuda OBJECT
  src/fused_kernel.cu
)
set_target_properties(canny_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(canny_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main executable: Adrian's orchestration + image I/O + stb_image impl.
add_executable(gpu_canny_demo
  src/main.cpp
  src/canny_pipeline.cpp
  src/stb_image_impl.cpp
  $<TARGET_OBJECTS:canny_cuda>
)
target_include_directories(gpu_canny_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${STB_DIR}
)
target_link_libraries(gpu_canny_demo PRIVATE CUDA::cudart)
