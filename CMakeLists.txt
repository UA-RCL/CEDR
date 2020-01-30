cmake_minimum_required(VERSION 3.10)
project(dssoc_emulator VERSION 1.0
                               DESCRIPTION "Linux userspace runtime for DAG-based applications and DSSoC emulation"
                               LANGUAGES CXX)

option(BuildARM "Configure build target for ARM-based systems (targets x86 by default)" FALSE)
option(UseLIEF "Include support for LIEF libraries and reading DAG JSON directly from instrumented application binaries (FALSE by default)" FALSE)
option(LIEF_Path "Path to compiled LIEF binaries if UseLIEF is selected" "")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER clang++)

# Flags that are used if CMake is told to generate a debug build (i.e. with -DCMAKE_BUILD_TYPE:STRING=DEBUG)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

if(BuildARM)
    add_executable(dssoc_emulator
            src/main.cpp
            src/application.cpp
            src/hardware.cpp
            src/dag_parse.cpp
            src/runtime.cpp
            src/scheduler.cpp
            src/config_parse.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -target aarch64-linux-gnu -DARM")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--export-dynamic")

    target_include_directories(dssoc_emulator PRIVATE /usr/aarch64-linux-gnu/include/c++/7/aarch64-linux-gnu)
    target_include_directories(dssoc_emulator PRIVATE /usr/aarch64-linux-gnu/include)
    target_include_directories(dssoc_emulator PRIVATE extern)
    target_include_directories(dssoc_emulator PRIVATE src/include)

    target_link_libraries(dssoc_emulator PRIVATE pthread)
    target_link_libraries(dssoc_emulator PRIVATE dl)
else()
    add_executable(dssoc_emulator
            src/main.cpp
            src/application.cpp
            src/hardware.cpp
            src/dag_parse.cpp
            src/runtime.cpp
            src/scheduler.cpp
            src/config_parse.cpp)

    target_include_directories(dssoc_emulator PRIVATE extern)
    target_include_directories(dssoc_emulator PRIVATE src/include)

    target_link_libraries(dssoc_emulator PRIVATE pthread)
    target_link_libraries(dssoc_emulator PRIVATE dl)
endif()

if (UseLIEF)
    if (${LIEF_Path} STREQUAL "")
        message(WARNING "UseLIEF was requested but the LIEF path was not set")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSELIEF")

        target_include_directories(dssoc_emulator PRIVATE ${LIEF_Path}/build/include)
        target_include_directories(dssoc_emulator PRIVATE ${LIEF_Path}/include)

        target_link_directories(dssoc_emulator PRIVATE ${LIEF_Path}/build)
        target_link_libraries(dssoc_emulator PRIVATE LIEF)
    endif()
    
endif()

# Enable additional targets for running clang-format/clang-tidy
# https://www.labri.fr/perso/fleury/posts/programming/using-clang-tidy-and-clang-format.html
file(GLOB_RECURSE
    ALL_CXX_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/src/*.[ch]pp)

# Add clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format-7")
if (CLANG_FORMAT)
    add_custom_target(
        clang-format
        COMMAND clang-format-7
        -i
        -style=file
        ${ALL_CXX_SOURCE_FILES}
    )
endif()

# Add clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy-7")
if (CLANG_TIDY)
    add_custom_target(
        clang-tidy
        COMMAND clang-tidy-7
        ${ALL_CXX_SOURCE_FILES}
        -config=''
        --
        -std=c++${CMAKE_CXX_STANDARD}
        -I${PROJECT_SOURCE_DIR}/extern
        -I${PROJECT_SOURCE_DIR}/src/include
    )
endif()