# API-Based CEDR - Getting Started

In this example workflow, we will familiarize ourselves with setting up our system to run API-based applications on CEDR

We will begin by first building and running on an x86_64 system with no accelerators, and then we will discuss execution on an aarch64-based ZCU102 FPGA with accelerators.

## Initial CEDR Compilation - x86_64

If you came from [README.md](README.md), this step might already be completed. 
Just in case, though, we'll start by building CEDR itself. 
Simply run the following from the repository root folder:

```bash
mkdir build
cd build
cmake ..
make -j $(nproc)
```

After this, we can go build our first application

## Radar Correlator Compilation

Navigate to [applications/APIApps/radar_correlator](applications/APIApps/radar_correlator) and open [radar_correlator.c](applications/APIApps/radar_correlator/radar_correlator.c).
Near the top of the program, you can see that we have `#include "dash.h"`.
This header file comes from [libdash/dash.h](libdash/dash.h), and it's how user applications know what API functions they can call.
Throughout the rest of the code, you can see that we call `DASH_FFT` three times -- twice as "forward" FFTs and once as an inverse "IFFT".
First, we'll build as a standard C application and check the expected behavior. 

Go ahead and run

```bash
make standalone
```

This will produce a file called `radar_correlator-x86.out`. Running it with `./radar_correlator-x86.out`, we should see something like:

```bash
[fft] Running a 512-Pt FFT on the CPU
[fft] Running a 512-Pt FFT on the CPU
[fft] Running a 512-Pt IFFT on the CPU
Radar correlator complete; lag Value is: -0.037000
```

From here, we can go ahead and build the CEDR-compatible version with
```bash
make
```

This produces `radar_correlator-x86.so` which is similar to the other file, but it has a different extension. 
It turns out this is because it's a _shared object_, a binary that is meant to be loaded as a part of another program. 
In our case, we'll be giving this binary to CEDR for it to run.
Copy this binary along with all output data to our CEDR build directory from earlier

```bash
# Assuming your CEDR build folder is in the repo root and named "build"
# Adjust as necessary
cp -r radar_correlator-x86.so ./input ../../../build 
```

After this, change to your CEDR build directory (i.e. `cd ../../../build`)
## Running API-based CEDR

As mentioned at the end of the [README.md](README.md), When running CEDR, there are a few key binaries: `cedr`, `sub_dag`, and `kill_daemon`.
Running `./cedr --help`, we can see the expected arguments:

```bash
An emulation framework for running DAG-based applications in linux userspace
Usage:
  CEDR [OPTION...]

  -l, --log-level arg    Set the logging level (default: INFO)
  -c, --config-file arg  Optional configuration file used to configure the
                         CEDR runtime. Defaults are chosen for all unspecified
                         values.
  -h, --help             Print help
```

We don't need to both with adjusting logging for now, but we can certainly provide a configuration file. One is provided by default in [daemon_config.json](daemon_config.json). For our initial testing, we will change `Loosen Thread Permissions` to `true` as it makes a few things easier.

In one terminal, we'll go ahead and launch CEDR with this updated configuration file:
```bash
./cedr -c ../daemon_config.json
```

Meanwhile, we will submit our application using `sub_dag`. Checking with `./sub_dag --help`:
```bash
A helper program for submitting applications to a daemon-based CEDR process
Usage:
  sub_dag [OPTION...]

  -a, --app-shared-obj arg  Specify a list of application shared object files
                            to invoke
  -n, --num-instances arg   For each application, specify the number of
                            instances to invoke
  -p, --periodicity arg     Specifies a periodic injection for instances
  -h, --help                Print help
```

We can see that we will go ahead and submit one instance with:
```bash
./sub_dag -a ./radar_correlator-x86.so -n 1
```

In our CEDR terminal, we can see that our application was received and executed, and near the end we can see our same output of `lag Value is: -0.037000`

```
2022-04-29 20:03:34.958 INFO  [3060664] [launchDaemonRuntime@537] Received application: radar_correlator-x86.so. Will attempt to inject 1 instances of it with a period of 0 microseconds
2022-04-29 20:03:34.958 INFO  [3060664] [parse_binary@45] I have not opened this shared object before. Looking for it at "./radar_correlator-x86.so"
2022-04-29 20:03:34.958 INFO  [3060664] [launchDaemonRuntime@639] [DEBUG_SETAFFINITY] Trying to set the first pthread affinity
2022-04-29 20:03:34.958 INFO  [3060664] [launchDaemonRuntime@663] Thread for application radar_correlator-x86 launched!
2022-04-29 20:03:34.966 INFO  [3060962] [enqueue_kernel@43] I am inside the runtime's codebase, unpacking my args to enqueue a new FFT task
2022-04-29 20:03:34.966 INFO  [3060962] [enqueue_kernel@80] I have finished initializing my FFT node, pushing it onto the task list
2022-04-29 20:03:34.966 INFO  [3060962] [enqueue_kernel@88] I have pushed a new task onto the work queue, time to go sleep until it gets scheduled and completed
2022-04-29 20:03:34.966 INFO  [3060664] [launchDaemonRuntime@673] Scheduling round found 1 tasks in ready task queue!
2022-04-29 20:03:34.966 INFO  [3060664] [attemptToAssignTaskToPE@57] Task pushed to the to-do queue of the requested resource
[fft] Running a 512-Pt FFT on the CPU
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@43] I am inside the runtime's codebase, unpacking my args to enqueue a new FFT task
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@80] I have finished initializing my FFT node, pushing it onto the task list
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@88] I have pushed a new task onto the work queue, time to go sleep until it gets scheduled and completed
2022-04-29 20:03:34.967 INFO  [3060664] [launchDaemonRuntime@673] Scheduling round found 1 tasks in ready task queue!
[fft] Running a 512-Pt FFT on the CPU
2022-04-29 20:03:34.967 INFO  [3060664] [attemptToAssignTaskToPE@57] Task pushed to the to-do queue of the requested resource
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@43] I am inside the runtime's codebase, unpacking my args to enqueue a new FFT task
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@80] I have finished initializing my FFT node, pushing it onto the task list
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@88] I have pushed a new task onto the work queue, time to go sleep until it gets scheduled and completed
2022-04-29 20:03:34.967 INFO  [3060664] [launchDaemonRuntime@673] Scheduling round found 1 tasks in ready task queue!
[fft] Running a 512-Pt IFFT on the CPU
2022-04-29 20:03:34.967 INFO  [3060664] [attemptToAssignTaskToPE@57] Task pushed to the to-do queue of the requested resource
Radar correlator complete; lag Value is: -0.037000
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@350] I am inside the runtime's codebase, injecting a poison pill to tell the host thread that I'm done executing
2022-04-29 20:03:34.967 INFO  [3060962] [enqueue_kernel@363] I have pushed the poison pill onto the task list
2022-04-29 20:03:34.967 INFO  [3060664] [launchDaemonRuntime@734] Number of resolved applications: 1; Number of killed applications: 0
```

To properly end the CEDR after this, we can kill it with `./kill_daemon`. This will generate a log file inside `log_dir/experiment0`. Taking a look at this file:

```bash
> cat ./log_dir/experiment0/timing_trace.log 
app_id: 0, app_name: radar_correlator-x86, task_id: 0, task_name: FFT, resource_name: cpu1, ref_start_time: 1478539080482297, ref_stop_time: 1478539080526960, actual_exe_time: 44663
app_id: 0, app_name: radar_correlator-x86, task_id: 1, task_name: FFT, resource_name: cpu2, ref_start_time: 1478539080747865, ref_stop_time: 1478539080778913, actual_exe_time: 31048
app_id: 0, app_name: radar_correlator-x86, task_id: 2, task_name: FFT, resource_name: cpu3, ref_start_time: 1478539080826201, ref_stop_time: 1478539080857209, actual_exe_time: 31008
```

We can see that all three FFT resources were scheduled to `cpu1`, `cpu2`, and `cpu3` respectively.

## CEDR Compilation - ZCU102

Moving on to the aarch64-based build for ZCU102 FPGA with accelerators. We'll start by building CEDR itself. This time we will use the [toolchain](toolchains/aarch64-linux-gnu.toolchain.cmake) file for cross-compilation.
Simply run the following from the repository root folder:

```bash
mkdir build-arm
cd build-arm
cmake -DLIBDASH_MODULES="FFT GEMM" --toolchain=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j $(nproc)
```

This will create an executable file for `cedr`, `sub_dag`, `kill_deamon`, and `libdash-rt` for aarch64 platforms. We can check the platform type of an executable using the `file` command:


```bash
file cedr
```
<pre>
cedr: ELF 64-bit LSB shared object, <b>ARM aarch64</b>, version 1 (GNU/Linux), dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, BuildID[sha1]=7374bd01c8ded1d48f9dd191e9010496bdffae34, for GNU/Linux 3.7.0, not stripped
</pre>

Since we also used the `-DLIBDASH_MODULES="FFT GEMM"` flag, we also enabled FFT and GEMM accelerator function calls for `DASH_FFT` and `DASH_GEMM` API calls. We can test if these functions are available or not by running the following commands:

```bash
nm -D libdash-rt/libdash-rt.so | grep -E '*_fft$|*_gemm$'
```
<pre>
0000000000006974 T <b>DASH_FFT_fft</b>
000000000000788c T <b>DASH_GEMM_gemm</b>
</pre>

After this, we can go to build our application using cross-compilation for aarch64

## Radar Correlator Cross-compilation

Assuming you came here after building the radar correlator for x86_64, we will directly move to compile the radar correlator for aarch64. First, navigate to [applications/APIApps/radar_correlator](applications/APIApps/radar_correlator) folder. Then run the following command to build the executable for aacrh64:

```bash
CC=aarch64-linux-gnu-gcc ARCH=aarch64 make
```

For source codes written in C++ we use `CXX=aarch64-linux-gnu-g++` and for source codes written in C we use `CC=aarch64-linux-gnu-gcc` to change the default compiler to the corresponding cross-compiler. The `ARCH=aarch64` is only used to change the output file's naming convention, which is `radar_correlator-<ARCH>.so` where `ARCH=x86` by default. We can also check the platform the shared object was created for by using the `file` command again.
```bash
file radar_correlator-aarch64.so
```
<pre>
radar_correlator-aarch64.so: ELF 64-bit LSB shared object, <b>ARM aarch64</b>, version 1 (SYSV), dynamically linked, BuildID[sha1]=0eeff7f86d35d1a1e4775432cd05476797f0a442, not stripped
</pre>

After verifying the file is compiled for the correct platform, copy the file and inputs to the build directory and change your working directory to the `build-arm` directory just like we did for x86_64:

```bash
# Assuming your CEDR build folder is in the repo root and named "build"
# Adjust as necessary
cp -r radar_correlator-aarch64.so ./input ../../../build-arm
cd ../../../build-arm
```
## Running API-based CEDR on ZCU102

Before going into the zcu102 first copy the [daemon_config.json](daemon_config.json) file to the `build-arm` directory. From the build-arm directory, run:

```bash
cp ../daemon_config.json ./
```

Now we will ssh into the ZCU102, and enter the password when prompted:

```bash
ssh <user-name>@<zcu102-ip>
``` 

IIf you like, you can create a folder for yourself on the board where you will be working for the remainder of this tutorial. Now create a folder in the desired working directory called `mnt` and create an sshfs connection for the `mnt` directory using the build folder (`build-arm`) on your local machine:

```bash
cd <desired workspace>
mkdir mnt
sshfs <user_name>@<ip for the localmachine>:<path to CEDR repository root folder>/build-arm mnt
cd mnt
```

After these steps are completed, if you type `ls` you should see all the files you had on the local machine is also here on the zcu102.

Before running CEDR, we need to enable FFT and GEMM accelerator in the [daemon_config.json](daemon_config.json) file and loosen the thread permission just like we did for x86_64. By the time this tutorial was written, we had 2 FFT and 2 GEMM accelerators available in the FPGA image. We can put any number between 0-2 to the corresponding fields on the [daemon_config.json](daemon_config.json) file. Change the file with the following `Worker Threads` setup:

```bash
"Worker Threads": {
        "cpu": 3,
        "fft": 2,
        "gemm": 2,
        "gpu": 0
    },
...
"Loosen Thread Permissions": true,
```
Execution of CEDR is the same as the x86_64 version. In one terminal launch CEDR:

```bash
./cedr -c ./daemon_config.json
```
After launching CEDR, you should see the function handles for FFT and GEMM accelerators are successfully grabbed.

In another terminal, we will submit two instances of `radar_correlator` using `sub_dag`:

```bash
./sub_dag -a ./radar_correlator-aarch64.so -n 2
```

On the first terminal where you ran the CEDR, you should see lines with `fft-0` and `fft-1`, indicating the FFT accelerators are used for the execution, and the output results are `lag Value is: -0.037000` for both instances. Now kill the CEDR by running `./kill_deamon` on the second terminal and check the resource_name fields for 6 FFT tasks:

```bash
cat ./log_dir/experiment0/timing_trace.log 
```
<pre>
app_id: 0, app_name: radar_correlator-aarch64, task_id: 0, task_name: FFT, resource_name: cpu1, ref_start_time: 869327327097343, ref_stop_time: 869327327306504, actual_exe_time: 209161  
app_id: 1, app_name: radar_correlator-aarch64, task_id: 0, task_name: FFT, resource_name: cpu2, ref_start_time: 869327328071080, ref_stop_time: 869327328244197, actual_exe_time: 173117  
app_id: 1, app_name: radar_correlator-aarch64, task_id: 1, task_name: FFT, resource_name: cpu3, ref_start_time: 869327328654308, ref_stop_time: 869327328848928, actual_exe_time: 194620  
app_id: 0, app_name: radar_correlator-aarch64, task_id: 1, task_name: FFT, resource_name: <b>fft1</b>, ref_start_time: 869327329063949, ref_stop_time: 869327329374580, actual_exe_time: 310631  
app_id: 0, app_name: radar_correlator-aarch64, task_id: 2, task_name: FFT, resource_name: <b>fft2</b>, ref_start_time: 869327330063329, ref_stop_time: 869327330333756, actual_exe_time: 270427  
app_id: 1, app_name: radar_correlator-aarch64, task_id: 2, task_name: FFT, resource_name: cpu1, ref_start_time: 869327330804663, ref_stop_time: 869327330952048, actual_exe_time: 147385
</pre>

We can see that all the resources available for FFT execution (`cpu1`, `cpu2`, `cpu3`, `fft1`, and `fft2`) are being used.

## CEDR Compilation - GPU

**Whoops this is the end of the tutorial go yell at whoever wrote it**
