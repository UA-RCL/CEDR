cmake_minimum_required(VERSION 3.2)
project(SAR_BP)

set(SOURCE_FILES BP_SAR.c)
set(LIBRARIES "m" "gsl" "gslcblas")

add_executable(SAR_BP ${SOURCE_FILES})
target_link_libraries(SAR_BP ${LIBRARIES})
