# GNU Radio C++ Flow Graph CMakeLists.txt
#
# Title: horn
# GNU Radio version: 3.8.2.0


cmake_minimum_required(VERSION 3.8)


include_directories(
    ${GNURADIO_ALL_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    $ENV{HOME}/.grc_gnuradio
)




set(GR_LIBRARIES
    boost_system
    gnuradio-blocks
    gnuradio-runtime
    gnuradio-pmt
    log4cpp
)

add_executable(horn horn.cpp)
target_link_libraries(horn ${GR_LIBRARIES})
