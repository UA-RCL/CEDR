INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_DASH dash)

FIND_PATH(
    DASH_INCLUDE_DIRS
    NAMES dash/api.h
    HINTS $ENV{DASH_DIR}/include
        ${PC_DASH_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    DASH_LIBRARIES
    NAMES gnuradio-dash
    HINTS $ENV{DASH_DIR}/lib
        ${PC_DASH_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/dashTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DASH DEFAULT_MSG DASH_LIBRARIES DASH_INCLUDE_DIRS)
MARK_AS_ADVANCED(DASH_LIBRARIES DASH_INCLUDE_DIRS)
