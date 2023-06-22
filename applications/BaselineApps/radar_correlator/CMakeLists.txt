cmake_minimum_required(VERSION 3.2)
project(radar_correlator)

set(SOURCE_FILES radar_correlator.c)
set(LIBRARIES "m" "gsl" "gslcblas")

add_executable(radar_correlator ${SOURCE_FILES})
target_link_libraries(radar_correlator ${LIBRARIES})
