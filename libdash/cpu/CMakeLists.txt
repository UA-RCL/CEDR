cmake_minimum_required(VERSION 3.18)
project(libdash_cpu VERSION 1.0
  DESCRIPTION "A libdash module for enabling CPU kernel support"
        LANGUAGES CXX)

message(STATUS "Expanding libdash sources to include support for CPU execution")

# Set the sources and include directories needed for CPU execution
file(GLOB LIBDASH_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(LIBDASH_CPU_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

# Append sources and include directories for the CPU files themselves
set(LIBDASH_SRCS ${LIBDASH_SRCS} ${LIBDASH_CPU_SRCS})
set(LIBDASH_INCLUDES ${LIBDASH_INCLUDES} ${LIBDASH_CPU_INCLUDES})
# Trim off any duplicates
list(REMOVE_DUPLICATES LIBDASH_SRCS)
list(REMOVE_DUPLICATES LIBDASH_INCLUDES)

# Set the updated values in the parent scope (in case we're called via add_subdirectory)
# https://stackoverflow.com/a/25217937
get_directory_property(hasParent PARENT_DIRECTORY)
if (hasParent)
  set(LIBDASH_SRCS ${LIBDASH_SRCS} PARENT_SCOPE)
  set(LIBDASH_INCLUDES ${LIBDASH_INCLUDES} PARENT_SCOPE)
endif()