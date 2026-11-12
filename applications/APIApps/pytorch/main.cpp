
#include <pybind11/embed.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <dlfcn.h>

#include "dash.h"

namespace py = pybind11;
void * python;

int main(int argc, char** argv){
    // printf("[C++] Attempting to run PyTorch model\n");

    // User should provide below or we should find it somehow
#if defined(ZCU102)
    python = dlopen("/usr/lib/libpython3.7m.so", RTLD_NOW | RTLD_GLOBAL);
#elif defined(X86)
    std::string str1 = "/usr/lib/x86_64-linux-gnu/libpython";
    std::string str2 = ".so";
    std::string libpython_path = str1 + PYTHON_VERSION + str2;
    python = dlopen(libpython_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
#elif defined(JETSON)
    std::string str1 = "/usr/lib/aarch64-linux-gnu/libpython";
    std::string str2 = ".so";
    std::string libpython_path = str1 + PYTHON_VERSION + str2;
    python = dlopen(libpython_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    // printf("%s\n", libpython_path.c_str());
#endif

    // printf("Python Version: %s\n", Py_GetVersion());
    // printf("Python Executable: %ls\n", Py_GetProgramFullPath());

    py::scoped_interpreter guard{};

    py::module_ sys = py::module_::import("sys");
    py::list sys_path = sys.attr("path");

#if defined(ZCU102)
    // Modify the last element
    if (!sys_path.empty()) {
        sys_path.attr("pop")();
        sys_path.append("../pyenv/lib/python3.7/site-packages");
    }
#elif defined(X86)
    // Modify the last element
    // We need first three to stay the same and then should append this as the fourth one. Sometimes there are more than 4 paths so we need to handle that
    if (!sys_path.empty()) {
        sys_path.attr("pop")();
	std::string path_str_append1 = "/lib/python";
	std::string path_str_append2 = "/site-packages";
	std::string path_str = PYTHON_ENV_PATH + path_str_append1 + PYTHON_VERSION + path_str_append2;
        sys_path.append(path_str);
    }
#elif defined(JETSON)
    // Modify the last element
    // We need first three to stay the same and then should append this as the fourth one. Sometimes there are more than 4 paths so we need to handle that
    if (!sys_path.empty()) {
        sys_path.attr("pop")();
	sys_path.append("torch_jetson/lib/python3.8/site-packages");
    }
#endif
    // py::print(sys.attr("path"));

    // if(argc == 0){
        py::module_ calc = py::module_::import("main");
    // } else{
        // std::string filename = std::string(argv[1]).substr(0, std::string(argv[1]).find_last_of('.'));
        // py::module_ calc = py::module_::import(filename.c_str());
    // }

    // printf("[C++] Terminating the C++ code after executing the python script\n");

}
