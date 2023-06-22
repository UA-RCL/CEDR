# CEDR - Compiler-integrated Extensible DSSoC Runtime
A linux userspace runtime framework for executing DAG-based/workflow-based applications

## Publications

See our list of publications on [the project website](https://ua-rcl.github.io/CEDR/#publications)

## Build dependencies and the Docker development container
If you are on a debian-based system or otherwise have `apt` available, all required packages for cross compilation/etc can be installed through `install_dependencies.sh`.

If you have [Docker](https://www.docker.com) available, a [development container](https://hub.docker.com/r/mackncheesiest/cedr_dev) can be setup that installs all prerequisites as required for both `x86` and `aarch64` cross-compilation development. 
The container can be built directly with
```bash
docker build --tag cedr_dev:latest .
```
And once it's complete, you can run the container from the repository root with the current working directory mounted as follows.
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

CEDR can be built to work in two modes: API-based and DAG-based. API-based CEDR is the more "modern" variation and, for new users, it is the approach we would recommend exploring. DAG-based CEDR is an older version, but it also supports some more advanced functionalities with regards to controlling precise parallelism and heterogeneous execution within your applications.

Depending on the build configuration desired, you can either build API-based or DAG-based by specifying `-DCEDR_TYPE=API` or `-DCEDR_TYPE=DAG` during your CMake configuration, respectively. By default, it builds the API-based approach if neither argument is specified.

If you would like to cross compile CEDR, assuming you have all necessary packages installed, that can be done using a toolchain defined in [toolchains](toolchains). To use an existing toolchain, simply specify it during your CMake configuration via `--toolchain=path/to/toolchain.cmake`. If a toolchain you need is not listed, we would recommend, at a minimum, adding a new toolchain file to the repository to handle it that others in the future can leverage. Additionally, you might consider (i) modifying [install_dependencies.sh](install_dependencies.sh) to perhaps handle automatically setting it up and (ii) and adding a quick compilation test for it to [.github/workflows/compile.yml](.github/workflows/compile.yml) so that future commits will test that they can at-least compile using this toolchain.

By default (unless you specify another build system), this will generate a makefile with standard targets (`make`, `make clean`, ...) as well as two custom targets: `make clang-format` and `make clang-tidy`. 
These options will invoke [clang-format](https://clang.llvm.org/docs/ClangFormat.html) and [clang-tidy](https://clang.llvm.org/extra/clang-tidy/) on the codebase, respectively.
Clang Format will reformat the codebase to be in line with the guidelines in [.clang-format](.clang-format), and likewise, Clang Tidy will run linter checks for common errors, bugs, etc with guidelines in [.clang-tidy](.clang-tidy)

### Miscellaneous CMake Options
There are a number of additional arguments that can be provided to CMake in order to influence the generated buildsystem:

---

Debug symbols can be included in the build by specifying the build type as DEBUG
```bash
cmake -DCMAKE_BUILD_TYPE:STRING=DEBUG ...
```

Alternatively, a binary that is built with optimization enabled and lacks debug symbols can be built with
```bash
cmake -DCMAKE_BUILD_TYPE:STRING=RELEASE ...
```

---

Support for runtime-based profiling of application nodes via PAPI can be enabled with
```bash
cmake -DUsePAPI ...
```

---

Support for enabling profiling within CEDR itself (i.e. to find bottlenecks) can be enabled with
```bash
cmake -DEnableProfiling ...
```
In which case a gcov report will be generated when it runs for post-execution analysis

## Getting Started

As mentioned above, there are two variations of CEDR: API-based and DAG-based. Within the scope of DAG-based CEDR, there are also two approaches: hand-crafted application compilation (in which case you manually convert your application into the DAG-based format that CEDR expects) and TraceAtlas-based application compilation (in which case we attempt to leverage [TraceAtlas](https://github.com/ruhrie/TraceAtlas) to automatically analyze and convert your application for use by the DAG-based CEDR).

Depending on the configuration desired, there are a few different guides that can be followed to get started, each in their own `GettingStarted.*.md` file.

- API-based CEDR: [GettingStarted.API.md](GettingStarted.API.md)
- Hand-crafted DAG-based CEDR: [GettingStarted.DAG.md](GettingStarted.DAG.md)
- TraceAtlas DAG-based CEDR: [GettingStarted.TraceAtlas.md](GettingStarted.TraceAtlas.md)

## Running CEDR

Regardless of the approach taken to build CEDR, each process produces effectively the same binaries and are run in the same way.

Across all approaches, the build creates three binaries: 
 1. `cedr`: The main binary to schedule incoming applications. It is intended to be executed as a background process.
 2. `sub_dag`: The command binary used to submit applications for execution.
 3. `kill_daemon`: The command binary used to safely terminate the `cedr` process. 

To submit an application to CEDR, begin by starting `cedr` in one shell:
```bash
./cedr [options]
```
Submit applications from another shell using:
```bash
./sub_dag -d /absolute/path/to/dag.json,/absolute/path/to/dag2.json -n [num_instances],[num_instances2]
```
Finally, when experiments are complete, terminate the daemon process using:
```bash
./kill_daemon
```
## Configuring the CEDR Daemon

The CEDR daemon can be configured using an optional JSON configuration file, with an example in the repo provided as `daemon_config.json`.
If no configuration file is specified, all values are initialized as specified in the ConfigManager [constructor](src-api/config_manager.cpp).
If a configuration file is used, then the values specified do not need to exhaustively configure the system -- any unspecified values will be initialized as specified in the constructor.
