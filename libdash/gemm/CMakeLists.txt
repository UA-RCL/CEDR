cmake_minimum_required(VERSION 3.18)
project(libdash_gemm VERSION 1.0
        DESCRIPTION "A libdash module for enabling GEMM accelerator support"
        LANGUAGES CXX)

message(STATUS "Expanding libdash sources to include support for GEMM accelerators")

if (NOT DEFINED COMMON_DMA_DIRECTORY)
  set(COMMON_DMA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../dma)
endif()

# Set the sources and include directories needed for GEMM
set(LIBDASH_GEMM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp ${COMMON_DMA_DIRECTORY}/dma.cpp)
set(LIBDASH_GEMM_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_DMA_DIRECTORY} ${DASH_TYPES_HEADER_DIR})

# Append sources and include directories for the GEMM files themselves
set(LIBDASH_SRCS ${LIBDASH_SRCS} ${LIBDASH_GEMM_SRCS})
set(LIBDASH_INCLUDES ${LIBDASH_INCLUDES} ${LIBDASH_GEMM_INCLUDES})
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__GEMM_ENABLE_MAIN")

add_executable(libdash_gemm_main ${LIBDASH_GEMM_SRCS})
target_include_directories(libdash_gemm_main PRIVATE ${LIBDASH_GEMM_INCLUDES})
set_target_properties(libdash_gemm_main PROPERTIES OUTPUT_NAME "gemm_main")