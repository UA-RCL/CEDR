cmake_minimum_required(VERSION 3.23)
project(libdash_gpu VERSION 1.0
  DESCRIPTION "A libdash module for enabling GPU support"
  LANGUAGES CXX CUDA)

message(STATUS "Expanding libdash sources to include support for GPU")

if (NOT DEFINED COMMON_DMA_DIRECTORY)
  set(COMMON_DMA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../dma)
endif()

# Set the sources and include directories needed for GPU
set(LIBDASH_GPU_SRCS 
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_fft.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_gemm.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_zip.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_conv2d.cu
  ${COMMON_DMA_DIRECTORY}/dma.cpp
)
set(LIBDASH_GPU_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_DMA_DIRECTORY} ${DASH_TYPES_HEADER_DIR})

# Append sources and include directories for the GPU files themselves
set(LIBDASH_SRCS ${LIBDASH_SRCS} ${LIBDASH_GPU_SRCS})
set(LIBDASH_INCLUDES ${LIBDASH_INCLUDES} ${LIBDASH_GPU_INCLUDES})
# Trim off any duplicates (since other modules may have, i.e., added DMA as well)
list(REMOVE_DUPLICATES LIBDASH_SRCS)
list(REMOVE_DUPLICATES LIBDASH_INCLUDES)

# Set the updated values in the parent scope (in case we're called via add_subdirectory)
# https://stackoverflow.com/a/25217937
get_directory_property(hasParent PARENT_DIRECTORY)
if (hasParent)
  set(LIBDASH_SRCS ${LIBDASH_SRCS} PARENT_SCOPE)
  set(LIBDASH_INCLUDES ${LIBDASH_INCLUDES} PARENT_SCOPE)
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
endif()

set(CMAKE_CUDA_COMPILE_OPTIONS_PIC -Xcompiler=-fPIC)

#set(CMAKE_CXX_FLAGS "--compiler-options '-fPIC' -lcufft ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-lcufft ${CMAKE_CXX_FLAGS}")

add_library(cufft SHARED IMPORTED)
set_target_properties(cufft PROPERTIES
   IMPORTED_LOCATION "/usr/local/cuda/lib64/libcufft.so"
   INTERFACE_INCLUDE_DIRECTORIES "/usr/local/cuda/include/"
)

add_executable(libdash_gpu_main ${LIBDASH_GPU_SRCS})
target_include_directories(libdash_gpu_main PRIVATE ${LIBDASH_GPU_INCLUDES})
set_target_properties(libdash_gpu_main PROPERTIES OUTPUT_NAME "gpu_main")
target_link_libraries(libdash_gpu_main cufft)
