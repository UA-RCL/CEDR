cmake_minimum_required(VERSION 3.2)
project(RF_convergence)

set(SOURCE_FILES RF_convergence_LU_projection.c)
set(LIBRARIES "m" "gsl" "gslcblas")

add_executable(RF_convergence ${SOURCE_FILES})
target_link_libraries(RF_convergence ${LIBRARIES})
