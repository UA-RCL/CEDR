cmake_minimum_required(VERSION 3.2)
project(pulse_doppler)

set(SOURCE_FILES pulse_doppler.c)
set(LIBRARIES "m" "gsl" "gslcblas")

add_executable(pulse_doppler ${SOURCE_FILES})
target_link_libraries(pulse_doppler ${LIBRARIES})
