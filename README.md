# CEDR - Compiler-integrated Extensible DSSoC Runtime

![Github CI](https://github.com/UA-RCL/CEDR/workflows/Compilation%20Checks/badge.svg)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/mackncheesiest/cedr_dev)

A linux userspace runtime framework for executing DAG-based/workflow-based applications on either x86 or ARM

Presentation from FOSDEM20: [https://fosdem.org/2020/schedule/event/llvm_aut_prog_het_soc/](https://fosdem.org/2020/schedule/event/llvm_aut_prog_het_soc/)

Paper from IPDPSW - HCW20: [https://arxiv.org/abs/2004.01636](https://arxiv.org/abs/2004.01636)

## Build dependencies and the Docker development container
If you are on a debian-based system or otherwise have `apt` available (developed on Ubuntu 18.04 LTS), all required packages for cross compilation/etc can be installed through `install_dependencies.sh`.
Note: this process involves installing LLVM 9 and it creates symlinks for `clang-9`, `clang++-9`, `clang-tidy-9`, and `clang-format-9` as `clang`, `clang++`, `clang-tidy`, and `clang-format` respectively.

If you have [Docker](https://www.docker.com) available, a [development container](https://hub.docker.com/repository/docker/mackncheesiest/cedr_dev) can be setup that installs all prerequisites as required for both `x86` and `aarch64` cross-compilation development. 
The container can be pulled with
```bash
docker pull mackncheesiest/cedr_dev:latest
```
Alternatively, the container can be built directly with
```bash
docker build --tag cedr_dev:latest .
```
And once it's complete, run the container from the repository root with the current working directory mounted as follows (prepending `mackncheesiest/` to the image name if using the image from the docker hub).
```bash
docker run -it --rm -v $(pwd):/root/repository cedr_dev:latest /bin/bash
```
With that in place, each of the following build configurations should work out of the box (and if they don't CI is failing 😄)

## Building

The build system of this project is [CMake](https://cmake.org/). With CMake installed, the project can be built through variations on the following:
```bash
mkdir build
cd build
cmake ${OPTIONS} ..
```

Depending on the build configuration desired, different options can be specified during CMake configuration. A brief table of configuration options is provided below:

|         |             Standalone          | Daemon                                 |
|---------|---------------------------------|----------------------------------------|
| `x86`     | `cmake ..`                      | `cmake -DBuildx86_Daemon:BOOL=TRUE ..` |
| `aarch64` | `cmake -DBuildARM:BOOL=TRUE ..` | `cmake -DBuildARM_Daemon:BOOL=TRUE ..` |

Sidenote: if you require cross compilation and cannot use the packages installed by `install_dependencies.sh` or the Docker container, the toolchain can be specified by updating the `AARCH64-LINUX-GNU-GCC` variable in `CMakeLists.txt`. 

By default (unless you specify another build system i.e. `ninja`), this will generate a makefile with standard targets (`make`, `make clean`, ...) as well as two custom targets: `make clang-format` and `make clang-tidy`. 
These options will invoke [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) on the codebase, respectively.
Clang Format will reformat the codebase to be in line with the guidelines in [.clang-format](.clang-format), and likewise, Clang Tidy will run linter checks for common errors, bugs, etc with guidelines in [.clang-tidy](.clang-tidy)

### Miscellaneous CMake Options
There are a number of additional arguments that can be provided to CMake in order to influence the generated buildsystem:

Debug symbols can be included in the build by specifying the build type as DEBUG
```bash
cmake -DCMAKE_BUILD_TYPE:STRING=DEBUG ...
```

Alternatively, a binary that is built with optimization enabled and lacks debug symbols can be built with
```bash
cmake -DCMAKE_BUILD_TYPE:STRING=RELEASE ...
```

Finally, on `x86` builds, experimental support is present for linking with the [LIEF](https://lief.quarkslab.com/) library and reading application DAGs that are embedded in executables using
```bash
cmake -DUseLIEF:BOOL=TRUE ...
```

## Running

An example set of `aarch64` and `x86` ELF binaries are provided in the `test_dag.tar.gz` archive.
Unzipping reveals `aarch64` and `x86` folders containing the respective JSON DAGs and shared object executables as well as an `input` folder containing input data for these applications.
When running in either configuration below, ensure that the `input` directory is located in the same directory as your `cedr` executable as executed applications will look for that relative path.
Also, be aware that on most systems, running with root privileges is required to set all of the required thread attributes. 

### Standalone
The "standalone" execution mode is useful for brief experiments or conducting simple application case studies.
It is invoked by building one of the `standalone` options from above and calling the resulting binary as

```bash
./cedr -c config.json [options] path/to/application/folder
```

The full list of options are available with `./cedr -h` or `./cedr --help`, and an example config file is shown below:
```json
{
  "validation_config": {
    "radar_correlator": 20
  },
  "performance_config": {
    "radar_correlator": {
      "period": 100,
      "probability": 0.6
    }
  },
  "trace_config": {
    "radar_correlator": 50,
    "radar_correlator": 75,
    "radar_correlator": 200
  }
}
```

In all cases, each application is specified by a name that corresponds with the name provided in its JSON file, but the data afterwards varies in meaning based on the configuration type.

- `"validation_config"`: each application is paired with a number of instances to run
- `"performance_config"`: each application is paired with an injection period in milliseconds and an injection probability that is compared against a uniform 0-1 random variable at each period
- `"trace_config"`: each application instance is paired with an absolute time in milliseconds relative to execution start when the application should be injected

### Daemon
The build for the daemon process creates three binaries: 
 1. `cedr`: The main binary to schedule incoming DAGs. It is executed as a background process.
 2. `sub_dag`: The command binary used to submit applications for execution.
 3. `kill_daemon`: The command binary used to safely terminate the `cedr` process. 

To use the daemon build, begin by starting `cedr` in one shell:
```bash
./cedr [options]
```
Submit applications from another shell using:
```bash
./sub_dag -d /absolute/path/to/dag.json -s /absolute/path/to/binary/folder/
```
Finally, when experiments are complete, terminate the daemon process using:
```bash
./kill_daemon
```
