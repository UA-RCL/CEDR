# CEDR - Compiler-integrated Extensible DSSoC Runtime
A Linux userspace runtime framework for executing DAG-based/workflow-based applications on either x86 or ARM

Presentation from FOSDEM20:
https://fosdem.org/2020/schedule/event/llvm_aut_prog_het_soc/

## Building
The main build system of this project is [CMake](https://cmake.org/). 
It requires CMake >= 3.10 as well as some variant of Clang/Clang++.
With CMake >= 3.10 installed, the project can be built through
```bash
mkdir build
cd build
cmake ..
```

This will generate a makefile with standard targets (`make`, `make clean`, ...) as well as two custom targets: `make clang-format` and `make clang-tidy`.

These options will invoke [clang-format-7](https://releases.llvm.org/7.0.0/tools/clang/docs/ClangFormat.html) and [clang-tidy-7](https://releases.llvm.org/7.0.0/tools/clang/tools/extra/docs/clang-tidy/index.html) on the codebase, respectively.
Clang Format will reformat the codebase to be in line with the guidelines in [.clang-format](.clang-format), and likewise, Clang Tidy will run linter checks for common errors, bugs, etc with guidelines in [.clang-tidy](.clang-tidy)

Currently, support is hardcoded for the LLVM 7.0 versions of these tools, so take note of this restriction if you run into issues along the lines of "clang-format-7 not found" (and feel free to improve it :) )

## CMake Options
There are a number of different arguments that can be provided to CMake in order to influence the final executable:

Debug symbols can be included in the build by specifying the build type as DEBUG
```bash
cmake -DCMake_BUILD_TYPE:STRING=DEBUG
```

Alternatively, a binary that is built with optimization enabled and lacks debug symbols can be built with
```bash
cmake -DCMAKE_BUILD_TYPE:STRING=RELEASE
```

Additionally, assuming an x86 host system, the default binary is built for x86. An ARM binary can be cross-compiled with 
```bash
cmake -DBuildARM:BOOL=TRUE
```

Finally, on x86 builds, experimental support is present for linking with the [LIEF](https://lief.quarkslab.com/) library and reading application DAGs from executables (rather than using a separate JSON file) using
```bash
cmake -DUseLIEF:BOOL=TRUE
```

## Running

The main usage information can be found by passing in the `-h/--help` flag.

```
./cedr --help

An emulation framework for running DAG-based applications in linux userspace
Usage:
  CEDR [OPTION...] [APP_DIRECTORY]

  -l, --log-level arg  Set the logging level (default: INFO)
  -m, --mode arg       Set the operational mode (default: VALIDATION)
  -s, --scheduler arg  Choose the scheduler to use (default: SIMPLE)
  -c, --config arg     File to load application configuration from 
                       (default: config.json)
  -h, --help           Print help
```

The only required argument is the positional application directory argument.
This directory is the location where shared objects and JSON-based DAGs will be parsed from.
See [test_app](applications/test_app) along with its associated Makefile for an example of such an application+JSON pair. 

All other arguments will attempt to use reasonable default values as listed in the help documentation.
A sample `config.json` is provided in the [repository root](config.json) as well as listed below.

```json
{
  "validation_config": {
    "test_app": 20
  },
  "performance_config": {
    "test_app": {
      "period": 100,
      "probability": 0.6
    }
  },
  "trace_config": {
    "test_app": 50,
    "test_app": 75,
    "test_app": 200
  }
}
```

## Problems

Please report issues and/or bugs to the standard Github [issue tracker](https://github.com/ua-rcl/CEDR/issues).

