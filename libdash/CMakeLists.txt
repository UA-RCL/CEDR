cmake_minimum_required(VERSION 3.18)
project(libdash VERSION 1.0
  DESCRIPTION "A library of functions for use with DASH applications"
  LANGUAGES CXX)

message(STATUS "Building libdash")

option(CEDR_BUILD "If building for use with CEDR, set this option to produce a shared object rather than statically linked library")

set(LIBDASH_SRCS)
set(LIBDASH_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})

find_library(GSL libgsl.a REQUIRED)
find_library(GSLCBLAS libgslcblas.a REQUIRED)

if (${CEDR_BUILD})
  message(STATUS "Building libdash for use with CEDR")
  
  set(ALL_LIBDASH_MODULES FFT GEMM GPU)
  set(LIBDASH_MODULES "" CACHE STRING "Hardware-acceleration modules to enable in libdash")
  set(CACHE LIBDASH_MODULES PROPERTY STRINGS ${ALL_LIBDASH_MODULES})

  set(LIBDASH_MODULES_LIST ${LIBDASH_MODULES})
  separate_arguments(LIBDASH_MODULES_LIST)

  set(COMMON_DMA_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dma")
  message(STATUS "DMA directory is: ${COMMON_DMA_DIRECTORY}")

  set(DASH_TYPES_HEADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "dash_types.h header directory is: ${DASH_TYPES_HEADER_DIR}")

  # Always include CPU support...
  add_subdirectory(cpu)

  foreach(MODULE IN LISTS LIBDASH_MODULES_LIST)
    if("${MODULE}" STREQUAL "GPU")
	    project(LANGUAGES CXX CUDA)
	    find_library(CUFFT libcufft.so PATHS "/usr/local/cuda/lib64/" REQUIRED)
    endif()
    if (NOT MODULE IN_LIST ALL_LIBDASH_MODULES)
      message(FATAL_ERROR "Module specified for libdash (${MODULE}) is not one of the accepted values (${ALL_LIBDASH_MODULES})")
    else()
      set(${MODULE}_ENABLED 1)
      set(${MODULE}_STANDALONE 0)
      message(STATUS "Enabling ${MODULE} module")
      # Our current convention is lowercase names for module folders
      string(TOLOWER "${MODULE}" MODULE_LOWERCASE)
      add_subdirectory(${MODULE_LOWERCASE})
    endif()
  endforeach()
else()
  message(STATUS "Building libdash for use as a standalone, CPU-only library")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCPU_ONLY")
  string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
  # Always include CPU support...
  add_subdirectory(cpu)
endif()

message(DEBUG "libdash srcs: ${LIBDASH_SRCS}")
message(DEBUG "libdash includes: ${LIBDASH_INCLUDES}")

if (${CEDR_BUILD})
  add_library(libdash-rt SHARED ${LIBDASH_SRCS})
  target_include_directories(libdash-rt PRIVATE ${LIBDASH_INCLUDES})
  target_link_libraries(libdash-rt PRIVATE ${GSL} PRIVATE ${GSLCBLAS} PRIVATE ${CUFFT})
  set_target_properties(libdash-rt PROPERTIES LIBRARY_OUTPUT_NAME dash-rt)
  install(TARGETS libdash-rt
          LIBRARY
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cedr)
else()
  add_library(dash_base OBJECT ${LIBDASH_SRCS})
  target_include_directories(dash_base PRIVATE ${LIBDASH_INCLUDES})

  add_custom_target(libdash
          ALL
	  COMMAND ${CMAKE_AR} rcsT libdash.a $<TARGET_OBJECTS:dash_base> ${GSL} ${GSLCBLAS} ${CUFFT}
          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          DEPENDS dash_base
          # Very important, otherwise the $<TARGET_OBJECTS:...> generator expression uses a semicolon-separated list and makes /bin/sh think two commands are present
          # Took forever to solve, but solved with: https://gitlab.kitware.com/cmake/cmake/-/issues/18205
          COMMAND_EXPAND_LISTS
          COMMENT "Combining all libdash objects into a single static archive"
          )
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libdash.a
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cedr)
endif()

# Regardless of which way we're building (for CEDR or not), 
# add a simple interface library target that ensures we get our headers populated in the right place
# 
# Check if this target is already defined as we typically "invoke" this CMakeLists multiple times
if (NOT (TARGET libdash-interface))
  add_library(libdash-interface INTERFACE)
  set_target_properties(libdash-interface 
    PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/dash.h;${CMAKE_CURRENT_SOURCE_DIR}/dash_types.h"
  )
  install(TARGETS libdash-interface
    PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cedr
  )
endif()