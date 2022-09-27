cmake_minimum_required(VERSION 3.18)
project(libdash_fft VERSION 1.0
        DESCRIPTION "A libdash module for enabling FFT accelerator support"
        LANGUAGES CXX)

message(STATUS "Expanding libdash sources to include support for FFT accelerators")

if (NOT DEFINED COMMON_DMA_DIRECTORY)
  set(COMMON_DMA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../dma)
endif()

# Set the sources and include directories needed for FFT
set(LIBDASH_FFT_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/fft.cpp ${COMMON_DMA_DIRECTORY}/dma.cpp)
set(LIBDASH_FFT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_DMA_DIRECTORY} ${DASH_TYPES_HEADER_DIR})

# Append sources and include directories for the FFT files themselves
set(LIBDASH_SRCS ${LIBDASH_SRCS} ${LIBDASH_FFT_SRCS})
set(LIBDASH_INCLUDES ${LIBDASH_INCLUDES} ${LIBDASH_FFT_INCLUDES})
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__FFT_ENABLE_MAIN")

add_executable(libdash_fft_main ${LIBDASH_FFT_SRCS})
target_include_directories(libdash_fft_main PRIVATE ${LIBDASH_FFT_INCLUDES})
set_target_properties(libdash_fft_main PROPERTIES OUTPUT_NAME "fft_main")
