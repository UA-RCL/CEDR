cmake_minimum_required(VERSION 3.10)
project(CEDR VERSION 1.0
        DESCRIPTION "Zynq-focused userspace runtime for DAG-based applications"
        LANGUAGES CXX)

option(BuildARM "Build a target for ARM-based systems (targets x86 by default)" FALSE)
option(BuildARM_Daemon "Build a target DAEMON for ARM-based systems" FALSE)
option(Buildx86_Daemon "Build a target DAEMON for x86-based systems" FALSE)
option(UseLIEF "Include support for LIEF libraries and reading DAG JSON directly from instrumented application binaries (FALSE by default)" FALSE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER clang++)
set(AARCH64-LINUX-GNU-GCC /usr/aarch64-linux-gnu)

# Flags that are used if CMake is told to generate a debug build (i.e. with -DCMAKE_BUILD_TYPE:STRING=DEBUG)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

if (BuildARM)
    add_executable(cedr
            src/main.cpp
            src/application.cpp
            src/hardware.cpp
            src/dag_parse.cpp
            src/runtime.cpp
            src/scheduler.cpp
            src/config_parse.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -target aarch64-linux-gnu -DARM")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--export-dynamic")

    target_include_directories(cedr PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7/aarch64-linux-gnu)
    target_include_directories(cedr PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/9/aarch64-linux-gnu)
    target_include_directories(cedr PRIVATE ${AARCH64-LINUX-GNU-GCC}/include)
    target_include_directories(cedr PRIVATE extern)
    target_include_directories(cedr PRIVATE src/include)

    target_link_libraries(cedr PRIVATE pthread)
    target_link_libraries(cedr PRIVATE dl)
    target_link_libraries(cedr PRIVATE rt)

elseif (BuildARM_Daemon)
    add_executable(cedr
            src/main_daemon.cpp
            src/application.cpp
            src/hardware.cpp
            src/dag_parse.cpp
            src/runtime.cpp
            src/scheduler.cpp
            src/config_parse.cpp
            src/ipc.cpp)
    add_executable(sub_dag
            src/ipc.cpp
            src/sub_dag.cpp)
    add_executable(kill_daemon
            src/ipc.cpp
            src/kill_daemon.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -target aarch64-linux-gnu -DARM -DDAEMON")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--export-dynamic")

    target_include_directories(cedr PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7/aarch64-linux-gnu)
    target_include_directories(cedr PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/9/aarch64-linux-gnu)
    target_include_directories(cedr PRIVATE ${AARCH64-LINUX-GNU-GCC}/include)
    target_include_directories(cedr PRIVATE extern)
    target_include_directories(cedr PRIVATE src/include)
    target_link_libraries(cedr PRIVATE pthread)
    target_link_libraries(cedr PRIVATE dl)
    target_link_libraries(cedr PRIVATE rt)

    target_include_directories(sub_dag PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7/aarch64-linux-gnu)
    target_include_directories(sub_dag PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/9/aarch64-linux-gnu)
    target_include_directories(sub_dag PRIVATE ${AARCH64-LINUX-GNU-GCC}/include)
    target_include_directories(sub_dag PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7.2.1/aarch64-linux-gnu/)
    target_include_directories(sub_dag PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7.2.1/)
    target_include_directories(sub_dag PRIVATE src/include)
    target_link_libraries(sub_dag PRIVATE pthread)
    target_link_libraries(sub_dag PRIVATE dl)
    target_link_libraries(sub_dag PRIVATE rt)

    target_include_directories(kill_daemon PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7/aarch64-linux-gnu)
    target_include_directories(kill_daemon PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/9/aarch64-linux-gnu)
    target_include_directories(kill_daemon PRIVATE ${AARCH64-LINUX-GNU-GCC}/include)
    target_include_directories(kill_daemon PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7.2.1/aarch64-linux-gnu/)
    target_include_directories(kill_daemon PRIVATE ${AARCH64-LINUX-GNU-GCC}/include/c++/7.2.1/)
    target_include_directories(kill_daemon PRIVATE src/include)
    target_link_libraries(kill_daemon PRIVATE pthread)
    target_link_libraries(kill_daemon PRIVATE dl)
    target_link_libraries(kill_daemon PRIVATE rt)
elseif (Buildx86_Daemon)
    add_executable(cedr
            src/main_daemon.cpp
            src/application.cpp
            src/hardware.cpp
            src/dag_parse.cpp
            src/runtime.cpp
            src/scheduler.cpp
            src/config_parse.cpp
            src/ipc.cpp)
    add_executable(sub_dag
            src/ipc.cpp
            src/sub_dag.cpp)
    add_executable(kill_daemon
            src/ipc.cpp
            src/kill_daemon.cpp)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDAEMON")
    target_include_directories(cedr PRIVATE extern)
    target_include_directories(cedr PRIVATE src/include)

    target_link_libraries(cedr PRIVATE pthread)
    target_link_libraries(cedr PRIVATE dl)
    target_link_libraries(cedr PRIVATE rt)

    target_include_directories(sub_dag PRIVATE src/include)
    target_link_libraries(sub_dag PRIVATE pthread)
    target_link_libraries(sub_dag PRIVATE dl)
    target_link_libraries(sub_dag PRIVATE rt)
    target_include_directories(kill_daemon PRIVATE src/include)
    target_link_libraries(kill_daemon PRIVATE pthread)
    target_link_libraries(kill_daemon PRIVATE dl)
    target_link_libraries(kill_daemon PRIVATE rt)

else ()
    add_executable(cedr
            src/main.cpp
            src/application.cpp
            src/hardware.cpp
            src/dag_parse.cpp
            src/runtime.cpp
            src/scheduler.cpp
            src/config_parse.cpp)

    target_include_directories(cedr PRIVATE extern)
    target_include_directories(cedr PRIVATE src/include)

    target_link_libraries(cedr PRIVATE pthread)
    target_link_libraries(cedr PRIVATE dl)
    target_link_libraries(cedr PRIVATE rt)
endif ()

if (UseLIEF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSELIEF")

    target_include_directories(cedr PRIVATE /localhome/jmack2545/rcl/DASH-SoC/llvm/LIEF/build/include)
    target_include_directories(cedr PRIVATE /localhome/jmack2545/rcl/DASH-SoC/llvm/LIEF/include)

    target_link_libraries(cedr PRIVATE LIEF)
endif ()

# Enable additional targets for running clang-format/clang-tidy
# https://www.labri.fr/perso/fleury/posts/programming/using-clang-tidy-and-clang-format.html
file(GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/*.[ch]pp
        ${PROJECT_SOURCE_DIR}/include/*.[ch]pp)

# Add clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if (CLANG_FORMAT)
    add_custom_target(
            clang-format
            COMMAND clang-format
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )
endif ()

# Add clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if (CLANG_TIDY)
    add_custom_target(
            clang-tidy
            COMMAND clang-tidy
            ${ALL_CXX_SOURCE_FILES}
            -config=''
            --
            -std=c++${CMAKE_CXX_STANDARD}
            -I${PROJECT_SOURCE_DIR}/extern
            -I${PROJECT_SOURCE_DIR}/include
            -I${PROJECT_SOURCE_DIR}/src/include
    )
endif ()
