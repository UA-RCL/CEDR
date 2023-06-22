# TraceAtlas + CEDR: Integration by Example

In this example workflow, we will go through all the steps involved in passing an application through the TraceAtlas analysis toolchain, software compilation flow, and deployment onto CEDR.

We will begin by first building and running on an x86_64 system with no accelerators, and then we will discuss execution on an aarch64-based ZCU102 FPGA with accelerators.

# X86_64

## Clone the repository

Clone [https://www.github.com/UA-RCL/CEDR_private](https://www.github.com/UA-RCL/CEDR_private) with all submodules
```bash
git clone --recurse-submodules https://www.github.com/UA-RCL/CEDR_private
```

## Install dependencies either on a given system or using docker container

With the repository cloned, to begin, install all necessary dependencies for CEDR and TraceAtlas development by either running `install_dependencies.sh` on an Ubuntu-based system or pulling the docker image with `docker pull mackncheesiest/cedr_dev:latest`.

## Build the x86 CEDR daemon

If using docker, start the docker container from the root CEDR directory with `docker run --rm -it -v $(pwd):/root/repository mackncheesiest/cedr_dev /bin/bash`. If not, no extra setup step is necessary. Run the following from the repository root:
```bash
mkdir build_x86
cd build_x86
cmake ..
cmake --build .
```

## Build TraceAtlas

Next, we need to build TraceAtlas and all software toolchain components via the following:
```bash
cd ${CEDR_ROOT}/TraceAtlas/vcpkg
./bootstrap-vcpkg.sh
./vcpkg install nlohmann-json spdlog indicators
cd ..
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../vcpkg/scripts/buildsystems/vcpkg.cmake ..
cmake --build .
```

## Trace test application for x86 execution

For this example, we are going to use temporal mitigation as our test application. 
This application is available in `applications/TraceAtlasApps/temporal_mitigation`.
Tracing is accomplished with the script `applications/traceExtractCompile.sh`.
`cd` into the `temporal_mitigation` application directory and run the following:
```bash
./do-trace.sh
```
This calls `traceExtractCompile.sh` under the hood and performs the code annotation, tracing, trace analysis, and final compilation for CEDR.
It produces output files of `temporal_mitigation-x86.json` and `temporal_mitigation-x86.so`.
We'll want to copy these into a folder that's easily accessible relative to your CEDR build directory such as `${CEDR_ROOT}/build_x86/apps`.
Additionally, you'll want to copy the `input` folder into `build_x86` directly as all relative file paths of the form `./input/*` are resolved relative to the directory of the parent cedr executable, not the shared object directory.

## Start CEDR daemon

With this, we have a CEDR-compatible application binary produced and are ready to execute. Begin by, in one terminal, starting the `cedr` process.
```bash
sudo ./cedr
```
Note that `sudo` is likely necessary in order to set all the required pthread attributes. If `sudo` is not available, then pass in a config file via the `--config-file` argument with `${CEDR_ROOT}/daemon_config.json` as a template, but set `"Loosen Thread Permissions"` to `true`.

## Submit generated application

With CEDR running, in a separate terminal, submit one instance of the application DAG via
```bash
sudo ./sub_dag -d apps/temporal_mitigation-x86.json -n 1
```

## Verify output

The stdout and stderr from running this application are shown in the standard output of the `cedr` binary.
This output should look something like the following:

```bash
2021-03-24 15:56:53.391 INFO  [2361] [launchDaemonRuntime@220] Received application: temporal_mitigation-x86.json. Will attempt to inject 1 instances of it with a period of 0 microseconds.
2021-03-24 15:56:53.393 INFO  [2361] [*parse_dag_and_binary@89] I have not opened this shared object before. Looking for it at "apps/temporal_mitigation-x86.so"
Temporal mitigation complete
2021-03-24 15:56:53.394 INFO  [2361] [launchDaemonRuntime@306] Completed the processing of input frame 0
```

With this execution complete, the main cedr process can be ended by running `sudo ./kill_daemon`.

# aarch64

With x86 execution verified, we can extend this example to running on an ARM-based platform with accelerators present.

## Build aarch64 CEDR daemon

Similar to building for x86, run the following from the repository root:
```bash
mkdir build_ARM
cd build_ARM
cmake --toolchain=${CEDR_ROOT}/toolchains/aarch64-linux-gnu.toolchain.cmake ..
cmake --build .
```

## Build MMULT kernel

As an added step on top of the previous example, we must build the matrix multiply kernel that will enable MMULT accelerator dispatch on the FPGA board.
```bash
cd kernels/MMULT
make aarch64
```
This produces an output binary in `./aarch64/mmult-aarch64.so`
It's worth noting here that whether or not this step requires any changes depends heavily on the FPGA configuration you'll be running on your ZCU102 board -- if the DMA and MMULT base addresses required match those from `kernels/include/dma.hpp` and `kernels/MMULT/mmult.h`, then you should be fine.
If you're not sure whether your board is configured correctly to execute the MMULT kernel, you can test it in a standalone fashion by running `make standalone_aarch64`.
This provides a standalone ELF binary that can be run on your board to test for functional correctness of your accelerator interface before then scaling up to using the accelerator in applications.

## Trace test application for aarch64 execution

The process for recompiling the application for aarch64 isn't difficult.
For accelerator support, we'll want to edit `do-trace.sh` to pass in `--semantic-opt` to the `traceExtractCompile.sh` script.
This will instruct the compiler to optimize the supported matrix multiplications in temporal mitigation such that we can also use the hardware accelerator.
After this, we again run the `do-trace.sh` script and override the architecture choice to be `aarch64`.
```bash
./do-trace.sh aarch64
```
Like previously, this generates `temporal_mitigation-aarch64.json` and `temporal_mitigation-aarch64.so` files.

## Transfer files to board
By some mechanism, transfer the resulting cedr, sub\_dag, temporal\_mitigation-aarch64.\*, mmult-aarch64.\*, and input/ files to your ZCU102 development board.
Here, we illustrate assuming the petalinux root user has an `apps` directory created in their home folder.

```bash
# CEDR binaries
scp build_ARM/{cedr,sub_dag,kill_daemon} root@zcu102:/home/root/
# MMULT accelerator binary
scp kernels/MMULT/aarch64/mmult-aarch64.so root@zcu102:/home/root/apps/mmult-aarch64.so
# Temporal mitigation DAG + binary
scp temporal_mitigation-aarch64.* root@zcu102:/home/root/apps/
# Temporal mitigation input
scp -r input/ root@zcu102:/home/root/
```

## Start cedr daemon
On one terminal connected to the ZCU102 development board, start the cedr daemon.

```bash
./cedr
```
## Submit generated application
In a separate terminal, submit the application DAG
```bash
./sub_dag -d apps/temporal_mitigation-aarch64.json -n 1
```

## Verify output
```bash
2021-03-24 23:05:38.985 INFO  [2562] [launchDaemonRuntime@220] Received application: temporal_mitigation-aarch64.json. Will attempt to inject 1 instances of it with a period of 0 microseconds
2021-03-24 23:05:39.002 INFO  [2562] [*parse_dag_and_binary@89] I have not opened this shared object before. Looking for it at "apps/temporal_mitigation-aarch64.so"
2021-03-24 23:05:39.011 INFO  [2562] [*parse_dag_and_binary@316] Node FuncCall_3 has platform mmult that uses a custom shared object
2021-03-24 23:05:39.011 INFO  [2562] [*parse_dag_and_binary@323] Looking for new shared object at "apps/mmult-aarch64.so"
[mmult] Initializing DMA buffers...
[dma] Initializing DMA 2 with control base address 0xa0010000
Resetting DMA 2
[dma] DMA 2 waiting for TX (MM2S) idle
[dma] Opening file descriptor to /dev/udmabuf1
[dma] Attempting to mmap
[dma] Opening file descriptor to /sys/class/u-dma-buf/udmabuf1/phys_addr
[mmult] Initialization complete!...
2021-03-24 23:05:39.012 INFO  [2562] [*parse_dag_and_binary@316] Node FuncCall_5 has platform mmult that uses a custom shared object
[dma] DMA 2 RX Status: "0x0"
[dma] DMA 2 RX Status: "0x0"
Temporal mitigation complete
2021-03-24 23:05:39.142 INFO  [2562] [launchDaemonRuntime@307] Completed the processing of input frame 0
```
