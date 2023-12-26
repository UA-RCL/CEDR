# API-Based CEDR Tutorial
In this tutorial, we will familiarize ourselves with setting up CEDR and perform following set of tutorials:
* Introducing an API call to an baseline C++ application.
* Design space exploration by varying number of compute resources across different scheduling heuristics in dynamically arriving workload scenarios.
* Integrating and evaluating scheduling heuristic with CEDR and conducting performance evaluation with dynamically arriving workload scenarios.
* Experiment with a new application that relies on key computation kernels such as FFT, GEMM, Convolution, Vector addition or Vector multiplication.
* Perform experiments on GPU and FPGA based SoCs.
* Integration of new API call to CEDR.

## 0. Requirements
* Linux machine or [docker image](https://hub.docker.com/r/mackncheesiest/cedr_dev/tags)
* CEDR Source Files: [CEDR repository for this tutorial](https://github.com/UA-RCL/CEDR/)

## 1. Initial Setup:
### Linux, Windows, and MAC
Install Docker based on the host machine platform using the [link](https://docs.docker.com/engine/install/#desktop).
Pull the existing latest [Docker container](https://hub.docker.com/r/mackncheesiest/cedr_dev/tags) with all dependencies installed.
Open a terminal and run the docker image using the following command:
```
docker run -it --rm -v <working-directory>:/root/repository mackncheesiest/cedr_dev:latest /bin/bash
```
Set <working-directory> as any folder in the host machine, all files will be stored here.
You can use this method to re-connect to the docker container after you exit.
Clone CEDR from GitHub using one of the following methods:
  * Clone with ssh:
```bash
git clone -b tutorial git@github.com:UA-RCL/CEDR.git
```
  * Clone with https:
```bash
git clone -b tutorial https://github.com/UA-RCL/CEDR.git
```

### Linux Specific (Requires root access)
Install git using the following command:
```bash
sudo apt-get install git-all
```
Clone CEDR from GitHub using one of the following methods:
  * Clone with ssh:
```bash
git clone -b tutorial git@github.com:UA-RCL/CEDR.git
```
  * Clone with https:
```bash
git clone -b tutorial https://github.com/UA-RCL/CEDR.git
```
Change your working directory to the cloned CEDR folder
```bash
cd CEDR
```
Using Docker Image:
* Install Docker using the following command:
```bash
sudo apt install docker.io
```
* Build the Docker image:
```bash
docker build --tag cedr_dev:latest .
```
* Run the Docker image:
```bash
docker run -it --rm -v $(pwd):/root/repository cedr_dev:latest /bin/bash
```

Without Using Docker Image:
* Install all the required dependencies using the following command (this will take some time):
```bash
sudo bash install_dependencies.sh
```

## 2. Building CEDR for x86

Navigate to the [root directory](./) and create a build folder
```bash
mkdir build
```
Change current directory to build
```bash
cd build
```
Call cmake and make to create CEDR executables for x86
```bash
cmake ../
make -j -$(nproc)
```
At this point there are 4 important files that should be compiled:
  * *cedr:* CEDR runtime daemon
  * *sub_dag:* CEDR job submission process
  * *kill_daemon:* CEDR termination process
  * *libdash-rt/libdash-rt.so:* Shared object used by CEDR for API calls

Look into [dash.h](libdash/dash.h) under [libdash](libdash) folder and see available API calls.

## 3. Introducing API Call to a Baseline C++ Application

### 3.1 Application Overview

Move to the Radar Correlator folder in the [applications](applications/APIApps/radar_correlator/) folder from [root directory](./)
```bash
cd applications/APIApps/radar_correlator
```

Look at the [non-API version](applications/APIApps/radar_correlator/radar_correlator_non_api.c) of the Radar Correlator and locate possible places for adding API calls to the application
  * Forward FFT call: Line 146
  * Forward FFT call: Line 167
  * Inverse FFT call: Line 194

Change the radar correlator to have DASH_FFT API calls and create a new file to place the API calls in the file
```bash
cp radar_correlator_non_api.c radar_correlator_fft.c
```
    
Make sure the [dash.h](libdash/dash.h) is included in the application
```C
#include "dash.h"
```

Change the `radar_correlator_non_api.c` to include `DASH_FFT` calls
```C
<gsl_fft_wrapper(fft_inp, fft_out, len, true);;
>DASH_FFT_flt(fft_inp, fft_out, len, true);

<gsl_fft_wrapper(fft_inp, fft_out, len, true);;
>DASH_FFT_flt(fft_inp, fft_out, len, true);

<gsl_fft_wrapper(fft_inp, fft_out, len, false);
>DASH_FFT_flt(fft_inp, fft_out, len, false);
```

Build radar correlator without API calls and observe the output
```bash
make non-api
./radar_correlator_non_api-x86.out
```

Build radar correlator with API calls (standalone execution outside CEDR) and compare the output against non-api version
```bash
make standalone
./radar_correlator_fft-x86.out
```

Build radar correlator shared object to be used with CEDR
```bash
make api
```

Copy the shared object and any input data files to the CEDR build folder
```bash
cp radar_correlator_fft-x86.so ../../../build
cp -r input/ ../../../build
``` 

### 3.2 CPU-based Preliminary Validation of the API Based Code

Move back to the CEDR build folder containing CEDR binaries
```bash
cd ../../../build
```

Copy the daemon configuration file from the [root repository](./) base directory to the build folder
```bash
cp ../daemon_config.json ./
```

Observe the contents of the `daemon_config.json`
  * *Worker Threads*: Specify the number of PEs for each resource
  * *Features*: Enable/Disable CEDR features
    * *Cache Schedules*: Enable/**Disable** schedule caching
    * *Enable Queueing*: **Enable**/Disable Queueing
    * *Use PAPI*: Enable/**Disable** PAPI based performance counters
    * *Loosen Thread Permissions*: Thread permission setup (**Enable**/Disable)
    * *Fixed Periodic Injection*: Inject jobs with fixed injection rate
    * *Exit When Idle*: Kill CEDR once the all running applications are terminated
  * *Scheduler*: Type of scheduler to use (SIMPLE, RANDOM, EFT, ETF, MET, or HEFT_RT)
  * *Random Seed*: Seed to be used on random operations
  * *DASH Binary Path*: List of paths to the libdash shared objects

Modify the `daemon_config.json` file to set the number of CPUs to 3 (or any other number).

<pre>
"Worker Threads": {
  "cpu": <b>3</b>,
  "fft": 0,
  "gemm": 0,
  "gpu": 0
},
</pre>

Run CEDR using the config file
```bash
./cedr -c ./daemon_config.json
```

Push CEDR to the background or open another terminal and navigate to the same build folder ([root repository](./)/build)
  * Option 1: Ctrl+z or run CEDR with `./cedr -c ./daemon_config.json &` in the previous step. Now this terminal can be used for running next steps.
  * Option 2: Open a second terminal and go to CEDR build directory, <code>cd  [root repository](./)/build</code>. Now this second terminal can be used for running next steps.

Run `sub_dag` to submit application(s) to CEDR

```bash
./sub_dag -a ./radar_correlator_fft-x86.so -n 1 -p 0
```

  * We use `-a` to specify the application shared object to be submitted
  * We use `-n` to specify the number of instances that are to be submitted
  * We use `-p` to specify the injection rate (waiting period between two instances in microsecond unit) when `-n` is more than 1
    * This is an optional argument, if not set it will be set to 0 by default

Look at the results and terminate CEDR using `kill_daemon`

```bash
./kill_daemon
```

Now, observe the log files generated in `log_dir/experiment0`. 
* `appruntime_trace.log` stores the execution time of the application.
* `schedule_trace.log` tracks the ready queue size and overhead of each scheduling decision.
* `timing_trace.log` stores the computing resource and execution time of the API call.

We can generate a Gantt chart showing the distribution of tasks to the processing elements. Navigate the `scripts/` folder from the [root directory](./) and run `gantt_k-nk.py` script.

```
cd scripts/
python3 gantt_k-nk.py ../build/log_dir/experiment0/timing_trace.log
```

Having built CEDR and compiled radar correlator application, we can proceed to performing design-space exploration now. 

## 4. Design Space Exploration

CEDR comes with some scripts that makes design-space exploration (DSE) rapid and easy. Now, we will go over the flow and define how to perform DSE step by step. First, navigate to folder where we accomodate API based CEDR scripts from [root directory](./).

```bash
cd scripts/scripts-API/run_scripts
```

We will initially run [daemon_generator.py](./scripts/scripts-API/run_scripts/daemon_generator.py) file to generate `daemon_config.json` files for our sweeps. We can modify the following code portion to denote scheduler types and hardware compositions. We set schedulers as `SIMPLE, ETF, and MET` while hardware compositions that are going to sweeped are picked as 4 CPUs at maximum since we don't have any accelerator on the x86 system. If there were any accelerator, we would also set the maximum number of accelerator that we would like to sweep up to. 

```python
SCHEDS = ["SIMPLE", "ETF", "MET"]
CPUS = 3
FFTS = 0
MMULTS = 0
ZIPS = 0
GPUS = 0
```

Then, we can see that number of each processing element starts from `0` all the way up to `maximum number of that processing element` looking at nested loops between Lines 26-31 (except CPU starts from 1). By changing the boundaries of the for loops, we can control the starting point of the sweep for each processing element. For this experiment, we will keep the file same.  

Next, we need to configure [run_cedr.sh](./scripts/scripts-API/run_scripts/run_cedr.sh) and [run_sub_dag.sh](./scripts/scripts-API/run_scripts/run_subdag.sh), which will concurrently run CEDR and submit applications. In `run_cedr.sh`, we need to set the following fields identical to the daemon config generator. Periodicity denotes the delay between injecting each application instance in microseconds.

```bash
declare -a SCHEDS=("SIMPLE" "MET" "ETF")
CPUS=3
FFTS=0
MMULTS=0
ZIPS=0
GPUS=0
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag
PERIODCOUNT=2
PERIODS=("1734" "2313")

declare -a WORKLOADS=("HIGH" )
```

In the case of `run_sub_dag.sh`, we need to set the following fields identical as before. In here, we also define `APPS` variable that stores the applications that we will sweep and `INSTS` variable that defines how many of each application will be submitted during each sweep. 

```bash
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "ETF")
declare -a CPUS=3
declare -a FFTS=0
declare -a MMULTS=0
declare -a ZIPS=0
declare -a GPUS=0
######################################################################

APPS=("radar_correlator_fft-x86.so")
INSTS=("5")

declare -a PERIODS=("1734" "2313")
PERIODCOUNT=2

declare -a WORKLOADS=("HIGH")
```

After getting everything ready, we can move scripts and configuration files to the [build](./build/) folder for starting the DSE. We also need to create a folder named `schedsweep_daemon_configs` in `build` folder to store configuration files. 

```bash
python3 daemon_generator.py
mkdir ../../../build/schedsweep_daemon_configs/
cp daemon_config*.json ../../../build/schedsweep_daemon_configs/
cp *.sh ../../../build
```

Navigate back to the `build` directory and remove all earlier log files in the `log_dir` directory.
```bash
cd ../../../build
rm -rf log_dir/*
``` 

Then, first execute `run_cedr.sh` for running CEDR with the DSE configurations on the first terminal. Then, pull up a new terminal and run `run_sub_dag.sh` for dynamically submitting the applications based on workload composition. If you are on Docker environment, execute the first ... commands on a separate terminal to be able to pull up a second terminal running docker container.

```bash
docker ps
# Take note of the number below "CONTAINER ID" column
docker exec -it <CONTAINER ID> /bin/bash
```

```bash
bash run_cedr.sh    # Execute on the first terminal
bash run_sub_dag.sh # Execute on the second terminal
```

After both scripts terminate, there should be a folder named `HIGH` in the `log_dir` containing as many files as there are trials. Each folder should have log files for each hardware composition, scheduler, and injection rate. To plot all the DSE results in a 3D format, first navigate to `scripts/scripts-API/` from [root directory](./).

```bash
cd scripts/scripts-API/
```

There are two scripts named `makedataframe.py` and `plt3dplot_inj.py` for plotting 3D diagram. For each DSE experiment, following lines in [makedataframe.py](scripts/scripts-API/makedataframe.py) should be modified.

```python
corelist = [' cpu1', ' cpu2', ' cpu3']  # Line 38

############# Edit parameters here ####################
# Starting from line 179
CPUS=3
FFTS=0
MMULTS=0
ZIPS=0

SCHEDS=["SIMPLE", "MET", "ETF"]

if WORKLOAD == 'HIGH':
    # Use following INJ_RATES and PERIODS for High latency workload data
    INJ_RATES=[10, 20]
    PERIODS=[1734, 2313]
elif WORKLOAD == 'LOW':
    print('Low workload is not specified for this setup')
else:
    print('Wrong workload type ', WORKLOAD, ' chosen, please choose either "HIGH" or "LOW"!')
    exit()

INJ_COUNT=int(args.injectionRateCount)
TRIALS=int(args.trial)
corelist = [' cpu1', ' cpu2', ' cpu3']  # Edit here

#######################################################
```

To learn about the input arguments of `makedataframe.py`, execute the script with `-h` option. Then, execute `makedataframe.py` script using the given arguments below for the DSE experiment in this tutorial. Other DSE experiments may require different set of input arguments. 

```bash
python3 makedataframe.py -h
python3 makedataframe.py -i ../../build/log_dir/ -w HIGH -o dataframe.csv -t 2 -r 2
```

Modify the following lines in the [plt3dplot_inj.py](scripts/scripts-API/plt3dplot_inj.py).

```python
### Configuration specification ###
### Starting from line 27
CPUS = 3
FFTS = 0
MMULTS = 0
ZIPS = 0
GPUS = 0
WORKLOAD = 'High'
TRIALS = 2
schedlist = {'SIMPLE':1, 'MET':2, 'ETF':3}
schedmarkerlist = {'SIMPLE':'o', 'MET':'o', 'ETF':'o'}
schednamelist = ['RR', 'MET', 'ETF']
```

Execute the script using the following commands. 

```bash
python3 plt3dplot_inj.py <input file name> <metric>
python3 plt3dplot_inj.py dataframe.csv CUMU  # Accumulates execution time of each API call
python3 plt3dplot_inj.py dataframe.csv EXEC  # Application execution time
python3 plt3dplot_inj.py dataframe.csv SCHED # Scheduling overhead
```

## 5. Integration and Evaluation of EFT Scheduler

Now navigate to [scheduler.cpp](scr-api/scheduler.cpp). This file contains various schedulers already tailored to work with CEDR. In this part of the tutorial, we will add the Earliest Finish Time(EFT) scheduler to CEDR. EFT heuristic schedules all the tasks in the `read queue` one by one based on the earliest expected finish time of the task on the available resources (processing elements -- PE). 

First, we will write the EFT scheduler as a C/C++ function. We will utilize the available variables for all the schedulers in CEDR. A list of the useful variables and their explanations can be found in the bulleted list below.

* cedr_config: Information about the current configuration of CEDR
* ready_queue: Tasks ready to be scheduled at the time of the scheduling event
* hardware_thread_handle: List of hardware threads that manages the available resources (PEs)
* resource_mutex: Mutex protection for the available resources (PEs)
* free_resource_count: Number of free resources (PEs) at the time of scheduling event, only useful if PE queues are disabled -- Not used in the tutorial

Based on the available variables, we will construct the prototype of the EFT scheduler as shown:

```C
int scheduleEFT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex, uint32_t &free_resource_count)
```

Leveraging the available variables, we will implement the EFT in C/C++ step by step. First, we will handle the initializations, then the main loop body for task-to-PE mapping based on the heuristic, followed by the actual PE assignment for the task, and end with final checks. After implementing the scheduler, we will add the EFT scheduler as one of the schedulers for CEDR and enable it in the runtime config.

### 5.1 Initialization

Here we will initialize some of the required fields for the EFT heuristic. We will use this function's start time as the reference current time while computing the expected finish time of a task on the PEs.

```C
  unsigned int tasks_scheduled = 0; // Number of tasks scheduled so far
  int eft_resource = 0; // ID of the PE that will be assigned to the task
  unsigned long long earliest_estimated_availtime = 0; // Estimated finish time initialization
  bool task_allocated; // Task assigned to PE successfully or not

  /* Get current time in nanosecond scale */
  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;

  long long avail_time; // Current available time of the PEs
  long long task_exec_time; // Estimated execution time of the task

  unsigned int total_resources = cedr_config.getTotalResources(); // Total Number of PEs available
```


### 5.2 EFT heuristic - Task-to-PE mapping

Here, we will have a double nested loop, where the outer loop will traverse all the tasks in the ready queue using the `ready_queue` variable. The inner loop will traverse all the PEs in the current runtime by indexing the `hardware_thread_handle`. 

```C
  // For loop to iterate over all tasks in Ready queue
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    earliest_estimated_availtime = ULLONG_MAX;
    // For each task, iterate over all PEs to find the earliest finishing one
    for (int i = total_resources - 1; i >= 0; i--) {
      auto resourceType = hardware_thread_handle[i].thread_resource_type; // FFT, ZIP, GEMM, etc.
      avail_time = hardware_thread_handle[i].thread_avail_time; // Based on estimated execution times of the tasks in the `todo_queue` of the PE
      task_exec_time = cedr_config.getDashExecTime((*itr)->task_type, resourceType); // Estimated execution time of the task
      auto finishTime = (curr_time >= avail_time) ? curr_time + task_exec_time : avail_time + task_exec_time; // estimated finish time of the task on the PE at i^th index
      auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resourceType]); // Check if the current PE support execution of this task
      /* Check if the PE supports the task and if the estimated finish time is earlier than what is found so far */
      if (resourceIsSupported && finishTime < earliest_estimated_availtime) {
        earliest_estimated_availtime = finishTime;
        eft_resource = i;
      }
    }
```

### 5.3 EFT heuristic - Task-to-PE assignment

Here, we will utilize a built-in function that does final checks before assigning the task to the given PE. Based on this assignment, it also handles the actual queue management and modification of any required field. Details of this function can be found [here](https://github.com/UA-RCL/CEDR/blob/tutorial/src-api/scheduler.cpp#L17-L64).

```C
    // Attempt to assign task on earliest finishing PE
    task_allocated = attemptToAssignTaskToPE(
      cedr_config, // Current configuration of the CEDR
      (*itr), // Task that is being scheduled
      &hardware_thread_handle[eft_resource], // PE that is mapped to the task based on the heuristic
      &resource_mutex[eft_resource], // Mutex protection for the PE's todo queue
      eft_resource // ID of the mapped PE
      );
```

### 5.4 Final checks

In the last part, we will check whether the task assignment to given PE was successful and move on to the next task in the `ready queue`.

```C
    if (task_allocated) { // If task allocated successfully
      tasks_scheduled++; // Increment the number of scheduled tasks
      itr = ready_queue.erase(itr); // Remove the task from ready_queue
      /* If queueing is disabled, decrement free resource count*/
      if (!cedr_config.getEnableQueueing()) {
        free_resource_count--;
        if (free_resource_count == 0)
          break;
      }
    } else { // If task is not allocated successfully
      itr++; // Go to the next task in ready_queue
    }
  }
  return tasks_scheduled;
```

### 5.5 Full EFT in C/C++

Now collecting all the steps, we will have the EFT function written in C/C++ that is tailored to CEDR, as shown:

```C
int scheduleEFT(ConfigManager &cedr_config, std::deque<task_nodes *> &ready_queue, worker_thread *hardware_thread_handle, pthread_mutex_t *resource_mutex, uint32_t &free_resource_count) {

  unsigned int tasks_scheduled = 0; // Number of tasks scheduled so far
  int eft_resource = 0; // ID of the PE that will be assigned to the task
  unsigned long long earliest_estimated_availtime = 0; // Estimated finish time initialization
  bool task_allocated; // Task assigned to PE successfully or not

  /* Get current time in nanosecond scale */
  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long curr_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;

  long long avail_time; // Current available time of the PEs
  long long task_exec_time; // Estimated execution time of the task

  unsigned int total_resources = cedr_config.getTotalResources(); // Total Number of PEs available

  // For loop to iterate over all tasks in Ready queue
  for (auto itr = ready_queue.begin(); itr != ready_queue.end();) {
    earliest_estimated_availtime = ULLONG_MAX;
    // For each task, iterate over all PEs to find the earliest finishing one
    for (int i = total_resources - 1; i >= 0; i--) {
      auto resourceType = hardware_thread_handle[i].thread_resource_type; // FFT, ZIP, GEMM, etc.
      avail_time = hardware_thread_handle[i].thread_avail_time; // Based on estimated execution times of the tasks in the `todo_queue` of the PE
      task_exec_time = cedr_config.getDashExecTime((*itr)->task_type, resourceType); // Estimated execution time of the task
      auto finishTime = (curr_time >= avail_time) ? curr_time + task_exec_time : avail_time + task_exec_time; // estimated finish time of the task on the PE at i^th index
      auto resourceIsSupported = ((*itr)->supported_resources[(uint8_t) resourceType]); // Check if the current PE support execution of this task
      /* Check if the PE supports the task and if the estimated finish time is earlier than what is found so far */
      if (resourceIsSupported && finishTime < earliest_estimated_availtime) {
        earliest_estimated_availtime = finishTime;
        eft_resource = i;
      }
    }

    // Attempt to assign task on earliest finishing PE
    task_allocated = attemptToAssignTaskToPE(
      cedr_config, // Current configuration of the CEDR
      (*itr), // Task that is being scheduled
      &hardware_thread_handle[eft_resource], // PE that is mapped to the task based on the heuristic
      &resource_mutex[eft_resource], // Mutex protection for the PE's todo queue
      eft_resource // ID of the mapped PE
      );

    if (task_allocated) { // If task allocated successfully
      tasks_scheduled++; // Increment the number of scheduled tasks
      itr = ready_queue.erase(itr); // Remove the task from ready_queue
      /* If queueing is disabled, decrement free resource count*/
      if (!cedr_config.getEnableQueueing()) {
        free_resource_count--;
        if (free_resource_count == 0)
          break;
      }
    } else { // If task is not allocated successfully
      itr++; // Go to the next task in ready_queue
    }
  }
  return tasks_scheduled;
}
```

### 5.6 Adding EFT as a scheduling option

Now, the only thing left is to ensure CEDR can run this function during scheduling events. To do this in the same [scheduler.cpp](scr-api/scheduler.cpp) file, we go to the end and update the [performScheduling](https://github.com/UA-RCL/CEDR/blob/tutorial/src-api/scheduler.cpp#L406) function. In the function where `sched_policy` is checked, we add another `else if` segment that checks whether the scheduling policy is `EFT`. If it is, we will call the function we just created.

```C
else if (sched_policy == "EFT") {
    tasks_scheduled += scheduleEFT(cedr_config, ready_queue, hardware_thread_handle, resource_mutex, free_resource_count);
  }
```

After adding EFT as one of the scheduling heuristic options to CEDR, we will need to rebuild CEDR in the `build` directory. First, navigate to [root directory](./), then follow the steps below to rebuild CEDR with EFT.

```bash
cd build
make -j
```

### 5.7 Enabling EFT for CEDR

In the [daemon_config.json](daemon_config.json) file, we updated the ["Scheduler"](https://github.com/UA-RCL/CEDR/blob/tutorial/daemon_config.json#L35) field to be "EFT" before running CEDR with the updated daemon config file.

```JSON
    "Scheduler": "EFT",
```

### 5.8 Running CEDR with EFT Scheduler

Using the same methods as in Section 3.2, we will run CEDR and see the use of the EFT scheduler. After running the following command, we will see that the scheduler to be used is selected as EFT in the displayed logs.

```bash
./cedr -c ./daemon_config.json -l VERBOSE | grep -E "(Scheduler|scheduler)" &
```

<pre>
[...] DEBUG [312918] [ConfigManager::parseConfig@136] Config contains key 'Scheduler', assigning config value to <b>EFT</b>
</pre>

After submitting the application with `sub_dag`, we will see that the newly added EFT scheduler is used during the scheduling event. 

```bash
./sub_dag -a ./radar_correlator_fft-x86.so -n 1 -p 0
```

<pre>
[...] DEBUG [312918] [performScheduling@475] Ready queue non-empty, performing task scheduling using <b>EFT</b> scheduler.
[...] DEBUG [312918] [performScheduling@475] Ready queue non-empty, performing task scheduling using <b>EFT</b> scheduler.
[...] DEBUG [312918] [performScheduling@475] Ready queue non-empty, performing task scheduling using <b>EFT</b> scheduler.
</pre>

Once everything is completed, we will terminate CEDR with `kill_daemon`.

```bash
./kill_daemon
```


## 6. Introducing a New API Call

In this section of the tutorial, we will demonstrate integration of a new API call to the CEDR. We will use `DASH_ZIP` API call as an example. 

Navigate to libdash folder from [root directory](./).
```bash
cd libdash
```

Add the API function definitions to the `dash.h`.

```C
void DASH_ZIP_flt(dash_cmplx_flt_type* input_1, dash_cmplx_flt_type* input_2, dash_cmplx_flt_type* output, size_t size, zip_op_t op);
void DASH_ZIP_flt_nb(dash_cmplx_flt_type** input_1, dash_cmplx_flt_type** input_2, dash_cmplx_flt_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);

void DASH_ZIP_int(dash_cmplx_int_type* input_1, dash_cmplx_int_type* input_2, dash_cmplx_int_type* output, size_t size, zip_op_t op);
void DASH_ZIP_int_nb(dash_cmplx_int_type** input_1, dash_cmplx_int_type** input_2, dash_cmplx_int_type** output, size_t* size, zip_op_t* op, cedr_barrier_t* kernel_barrier);
```

There 4 different function definitions here:
  1. DASH_ZIP_flt: Supports blocking ZIP calls for `dash_cmplx_flt_type`
  2. DASH_ZIP_flt_nb: Supports non-blocking ZIP calls for `dash_cmplx_flt_type`
  3. DASH_ZIP_int: Supports blocking ZIP calls for `dash_cmplx_int_type`
  4. DASH_ZIP_int_nb: Supports non-blocking ZIP calls for `dash_cmplx_int_type`

Add supported ZIP operation enums to `dash_types.h`
```C
typedef enum zip_op {
  ZIP_ADD = 0,
  ZIP_SUB = 1,
  ZIP_MULT = 2,
  ZIP_DIV = 3
} zip_op_t;
```

Add CPU implementation of the ZIP to [libdash/cpu](libdash/cpu/) `zip.cpp`. For simplicity, we just copy the original implementation.
```bash
cp original_files/zip.cpp libdash/cpu/
```

In `zip.cpp`, we also have the `enqueue_kernel` call in the API definition, which is how the task for this API will be sent to CEDR. A prototype of the `enqueue_kernel` function is given in line 12, and `enqueue_kernel` is used in non-blocking versions of the function (lines 83 and 113). The prototype is the same for all the APIs created for CEDR. The first argument has to be the function name, the second argument has to be the precision to be used, and the third argument shows how many inputs are needed for the calling function (for ZIP, this is 6). Now let's look at the ZIP-specific `enqueue_kernel` call.

```C
enqueue_kernel("DASH_ZIP", "flt", 6, input_1, input_2, output, size, op, kernel_barrier);
```

In this sample `enqueue kernel` call, we have 4 important portions:
  * ***"DASH_ZIP"***: Name of the API call
  * ***"flt"***: Type of the inputs on the API call
  * ***6***: Number of variables for the API call
  * Variables:
    * ***input_1***: First input of the ZIP
    * ***input_2***: Second input of the ZIP
    * ***output***: Output of the ZIP
    * ***size***: Array size for inputs and output
    * ***op***: ZIP operation type (ADD, SUB, MUL, or DIV)
    * ***kernel_barrier***: Contains configuration information of barriers for blocking and non-blocking implementation

In the `zip.cpp`, we need to fill the bodies of the four function definitions that are used so the application will call `enqueue_kernel` properly and hand off the task to CEDR for scheduling:

1. DASH_ZIP_flt
2. DASH_ZIP_flt_nb
3. DASH_ZIP_int
4. DASH_ZIP_int_nb

We also implement two more functions, which contains implementation of CPU-based ZIP operations. Functions are created with `_cpu` suffix so that CEDR can identify the functions correctly for CPU execution:

1. DASH_ZIP_flt_cpu: `dash_cmplx_flt_type`
2. DASH_ZIP_int_cpu: `dash_cmplx_int_type`

Having included API implementation, we should introduce the new API call to the system by updating CEDR header file ([./src-api/include/header.hpp](src-api/include/header.hpp)):

<pre>
enum api_types {DASH_FFT = 0, DASH_GEMM = 1, DASH_FIR = 2, DASH_SpectralOpening = 3, DASH_CIC = 4, DASH_BPSK = 5, DASH_QAM16 = 6, DASH_CONV_2D = 7, DASH_CONV_1D = 8, <b>DASH_ZIP = 9,</b> NUM_API_TYPES = <b>10</b>};

static const char *api_type_names[] = {"DASH_FFT", "DASH_GEMM", "DASH_FIR", "DASH_SpectralOpening", "DASH_CIC", "DASH_BPSK", "DASH_QAM16", "DASH_CONV_2D", "DASH_CONV_1D"<b>, "DASH_ZIP"</b>};
...
static const std::map<std::string, api_types> api_types_map = {{api_type_names[api_types::DASH_FFT], api_types::DASH_FFT},
                                                               {api_type_names[api_types::DASH_GEMM], api_types::DASH_GEMM},
                                                               {api_type_names[api_types::DASH_FIR], api_types::DASH_FIR},
                                                               {api_type_names[api_types::DASH_SpectralOpening], api_types::DASH_SpectralOpening},
                                                               {api_type_names[api_types::DASH_CIC], api_types::DASH_CIC},
                                                               {api_type_names[api_types::DASH_BPSK], api_types::DASH_BPSK},
                                                               {api_type_names[api_types::DASH_QAM16], api_types::DASH_QAM16},
                                                               {api_type_names[api_types::DASH_CONV_2D], api_types::DASH_CONV_2D},
                                                               {api_type_names[api_types::DASH_CONV_1D], api_types::DASH_CONV_1D},
                                                              <b>{api_type_names[api_types::DASH_ZIP], api_types::DASH_ZIP}</b>};
</pre>

Navigate to the build folder, re-generate the files, and check the `libdash-rt.so` shared object to verify the new ZIP-based function calls.
```bash
cd ../build
cmake ..
make -j $(nproc)
nm -D libdash-rt/libdash-rt.so | grep -E '*_ZIP_*'
```

To verify, build lane detection application which utilizes ZIP API calls.
```bash
cd applications/APIApps/lane_detection/
make fft_nb.so
cp track_nb.so image.png ../../../build/
```

Now, launch CEDR and submit lane detection application.
```bash
cd ../../../build/
rm -rf log_dir/*
./cedr -c ./daemon_config.json -l NONE &
./sub_dag -a ./track_nb.so -n 1 -p 0
```

Let's check the `timing_trace.log` for ZIP API calls.
```bash
cat log_dir/experiment0/timing_trace.log | grep -E '*ZIP*'
```

If you have a C++ based serial implementation of key kernels in your application, you can add your API call following the explanations in this section and replace your C++ kernel code with newly introduced API call following the Section 3.

## 7. Running Multiple Applications with CEDR on x86

### 7.1 Compilation of Applications
In this section, we will demonstrate CEDR's ability to manage dynamically arriving applications. Assuming you already have built CEDR following the previous steps, we will directly delve into compiling and running two new applications that are lane detection and pulse doppler.

Firstly, navigate to lane detection folder from the root folder of the repository, compile it for x86 and move shared object and input files to `build` folder for running with CEDR. (If you have already done this for the previous section, you don't have to compile the lane detection once more.)

```bash
cd applications/APIApps/lane_detection/
make fft_nb.so
cp track_nb.so image.png ../../../build
```

Now, let's do the above steps for pulse doppler application and create a output directory in `build` for writing its output. 

```bash
cd ../pulse_doppler/
make nonblocking
cp -r pulse_doppler-nb-x86.so input/ ../../../build
mkdir ../../../build/output
```

### 7.2 Running the Applications with CEDR

Now that we have all binaries and input files ready, we can proceed with running these applications with CEDR. First, navigate to `build` directory.

```bash
cd ../../../build
```

Then, launch CEDR with your desired configuration and submit both applications with varying number of instances and injection rates. 

```bash
./cedr -c ./daemon_config.json -l NONE &
./sub_dag -a ./track_nb.so,./pulse_doppler-nb-x86.so -n 1,5 -p 0,100
```

Observe the [output image of lane detection](./build/output_fft.png) and the [shift and time delay](./build/output/pulse_doppler_output.txt) calculated by pulse doppler.

## 8. GPU Based SoC Experiment (Nvidia Jetson AGX Xavier)

### 8.1 Building CEDR
Firstly, we need to connect to the Nvidia Jetson board through ssh connection. 
```bash
ssh <user-name>@<jetson-ip>
``` 

Clone CEDR from GitHub using one of the following methods <b>(we need to add one more line that's checking out to tutorial branch)</b>:
  * Clone with ssh:
```bash
git clone -b tutorial git@github.com:UA-RCL/CEDR.git
```
  * Clone with https:
```bash
git clone -b tutorial https://github.com/UA-RCL/CEDR.git
```

Then, we can build CEDR. Cross compiler is not necessary in this case since we are already logged into the host machine. We enable `GPU` flag for the CMake to be able to make use of the GPU as an accelerator. 
```bash
mkdir build
cd build
cmake -DLIBDASH_MODULES="GPU" ../
make -j -$(nproc)
```

### 8.2 Compilation

Now, we can compile the lane detection application running the following commands. This creates an executable shared object that we will move to `build` folder along with input image to run the `lane_detection`:
```bash
cd ../applications/APIApps/lane_detection/
make fft_nb.so
cp track_nb.so image.png ../../../build/
```

Also, let's compile the pulse doppler application running the following commands and move shared object and input to 'build' folder:
```bash
cd ../pulse_doppler/
make nonblocking
cp -r pulse_doppler-nb-x86.so input/ ../../../build/
```

### 8.3 Running the Applications with CEDR

Now, we need to go back to `build` folder and move `daemon_config.json` to it. Also, create a `output` folder for storing `pulse doppler` output:
```bash
cd ../../../build/
cp ../daemon_config.json ./
mkdir output/
```

We should enable `GPU` in `daemon_config.json` so that runtime will use it as a computing resource.

```JSON
"Worker Threads": {
        "cpu": 7,
        "fft": 0,
        "gemm": 0,
        "gpu": 1
    },
...
"Loosen Thread Permissions": true,
```

Execution of CEDR is the same as the x86_64 version. In one terminal launch CEDR:

```bash
./cedr -c ./daemon_config.json -l NONE &
```

In another terminal, we will submit an instance of `lane_detection` and five instances of `pulse doppler` using `sub_dag`:

```bash
./sub_dag -a ./track_nb.so,./pulse_doppler-nb-x86.so -n 1,5 -p 0,100
```

Now kill the CEDR by running `./kill_deamon` on the second terminal and check the resource_name fields for the first 5 FFT tasks:

```bash
head -n 10 ./log_dir/experiment0/timing_trace.log
```

<pre>
app_id: 3, app_name: track_nb, task_id: 0, task_name: DASH_FFT, resource_name: <b>cpu1</b>, ref_start_time: 15215236842193792, ref_stop_time: 15215236873412128, actual_exe_time: 31218336
app_id: 3, app_name: track_nb, task_id: 1, task_name: DASH_FFT, resource_name: <b>cpu2</b>, ref_start_time: 15215236842244288, ref_stop_time: 15215236873341824, actual_exe_time: 31097536
app_id: 3, app_name: track_nb, task_id: 2, task_name: DASH_FFT, resource_name: <b>cpu3</b>, ref_start_time: 15215236842259136, ref_stop_time: 15215236873367616, actual_exe_time: 31108480
app_id: 3, app_name: track_nb, task_id: 3, task_name: DASH_FFT, resource_name: <b>cpu4</b>, ref_start_time: 15215236842278592, ref_stop_time: 15215236873329504, actual_exe_time: 31050912
app_id: 3, app_name: track_nb, task_id: 4, task_name: DASH_FFT, resource_name: <b>cpu5</b>, ref_start_time: 15215236842303616, ref_stop_time: 15215236873309984, actual_exe_time: 31006368
app_id: 3, app_name: track_nb, task_id: 7, task_name: DASH_FFT, resource_name: <b>gpu1</b>, ref_start_time: 15215236842359360, ref_stop_time: 15215236878110208, actual_exe_time: 35750848
app_id: 3, app_name: track_nb, task_id: 5, task_name: DASH_FFT, resource_name: <b>cpu6</b>, ref_start_time: 15215236842560224, ref_stop_time: 15215236873353824, actual_exe_time: 30793600
app_id: 3, app_name: track_nb, task_id: 6, task_name: DASH_FFT, resource_name: <b>cpu7</b>, ref_start_time: 15215236843057888, ref_stop_time: 15215236873386784, actual_exe_time: 30328896
app_id: 3, app_name: track_nb, task_id: 11, task_name: DASH_FFT, resource_name: <b>cpu4</b>, ref_start_time: 15215236873349088, ref_stop_time: 15215236873802944, actual_exe_time: 453856
app_id: 3, app_name: track_nb, task_id: 9, task_name: DASH_FFT, resource_name: <b>cpu2</b>, ref_start_time: 15215236873349440, ref_stop_time: 15215236873780032, actual_exe_time: 430592
</pre>

Lastly, let's compare the outputs with x86 results to validate the correctness.

```bash
xdg-open output_fft.png
cat output/pulse_doppler_output.txt
```

## 9. FPGA Based SoC Experiment (ZCU102 MPSoC)

(Conv2d (accelerator) is not included in HCW release.)
Moving on to the aarch64-based build for ZCU102 FPGA with accelerators. We'll start by building CEDR itself. This time we will use the [toolchain](toolchains/aarch64-linux-gnu.toolchain.cmake) file for cross-compilation. If you are on Ubuntu 22.04, the toolchain requires running inside the docker container. (Self note: Be careful about platform.h)
Simply run the following commands from the repository root folder:

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

### 9.1 Cross-compilation

Assuming you came here after building the lane detection for x86_64, we will directly move to compile the lane detection for aarch64. First, navigate to [applications/APIApps/lane_detection](applications/APIApps/lane_detection) folder. Then run the following command to build the executable for aacrh64:

```bash
cd applications/APIApps/lane_detection
ARCH=aarch64 make fft_nb.so
file track_nb.so
```
<pre>
track_nb.so: ELF 64-bit LSB shared object, <b>ARM aarch64</b>, version 1 (SYSV), dynamically linked, BuildID[sha1]=fc95ba8be71bfb2f90164848211b62325f087007, not stripped
</pre>

After verifying the file is compiled for the correct platform, copy the file and inputs to the build directory:

```bash
# Assuming your CEDR build folder is in the root directory and named "build-arm"
cp track_nb.so image.png ../../../build-arm
```

Simply, perform the same operations for pulse doppler application:

```bash
cd ../pulse_doppler
ARCH=aarch64 make nonblocking
cp -r pulse_doppler-nb-aarch64.so input/ ../../../build-arm
```

### 9.2 Running API-based CEDR on ZCU102

Now, change your working directory to the `build-arm` directory. Before going into the zcu102 first copy the [daemon_config.json](daemon_config.json) file to the `build-arm` directory and create an output folder. From the build-arm directory, run:

```bash
cd ../../../build-arm
cp ../daemon_config.json ./
mkdir output/
```

Now we will ssh into the ZCU102, and enter the password when prompted:

```bash
ssh <user-name>@<zcu102-ip>
``` 

If you like, you can create a folder for yourself on the board where you will be working for the remainder of this tutorial. Now create a folder in the desired working directory called `mnt` and create an sshfs connection for the `mnt` directory using the build folder (`build-arm`) on your local machine:

```bash
cd <desired workspace>
mkdir mnt
sshfs <user_name>@<ip for the localmachine>:<path to CEDR repository root folder>/build-arm mnt
cd mnt
```

After these steps are completed, if you type `ls` you should see all the files you had on the local machine is also here on the zcu102.

Before running CEDR, we need to enable FFT and GEMM accelerator in the [daemon_config.json](daemon_config.json) file and loosen the thread permission just like we did for x86_64. By the time this tutorial was written, we had 2 FFT and 2 GEMM accelerators available in the FPGA image. We can put any number between 0-2 to the corresponding fields on the [daemon_config.json](daemon_config.json) file. Change the file with the following `Worker Threads` setup:

```json
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
./cedr -c ./daemon_config.json -l NONE &
```
After launching CEDR, you should see the function handles for FFT and GEMM accelerators are successfully grabbed.

In another terminal, we will submit an instance of `lane_detection` and five instances of `pulse doppler` using `sub_dag`:

```bash
./sub_dag -a ./track_nb.so,./pulse_doppler-nb-aarch64.so -n 1,5 -p 0,100
```

Now kill the CEDR by running `./kill_deamon` on the second terminal and check the resource_name fields for the first 10 FFT tasks:

```bash
head -n 10 ./log_dir/experiment0/timing_trace.log
```
<pre>
app_id: 0, app_name: track_nb, task_id: 0, task_name: DASH_FFT, resource_name: <b>cpu1</b>, ref_start_time: 5248428403449098, ref_stop_time: 5248428420053817, actual_exe_time: 16604719
app_id: 0, app_name: track_nb, task_id: 1, task_name: DASH_FFT, resource_name: <b>cpu2</b>, ref_start_time: 5248428408909254, ref_stop_time: 5248428420265209, actual_exe_time: 11355955
app_id: 0, app_name: track_nb, task_id: 3, task_name: DASH_FFT, resource_name: <b>fft1</b>, ref_start_time: 5248428408932076, ref_stop_time: 5248428420061538, actual_exe_time: 11129462
app_id: 0, app_name: track_nb, task_id: 4, task_name: DASH_FFT, resource_name: <b>fft2</b>, ref_start_time: 5248428409129026, ref_stop_time: 5248428420069309, actual_exe_time: 10940283
app_id: 0, app_name: track_nb, task_id: 2, task_name: DASH_FFT, resource_name: <b>cpu3</b>, ref_start_time: 5248428409170040, ref_stop_time: 5248428420067799, actual_exe_time: 10897759
app_id: 0, app_name: track_nb, task_id: 5, task_name: DASH_FFT, resource_name: <b>cpu1</b>, ref_start_time: 5248428420067189, ref_stop_time: 5248428420236706, actual_exe_time: 169517
app_id: 0, app_name: track_nb, task_id: 7, task_name: DASH_FFT, resource_name: <b>cpu3</b>, ref_start_time: 5248428420072569, ref_stop_time: 5248428420260798, actual_exe_time: 188229
app_id: 0, app_name: track_nb, task_id: 9, task_name: DASH_FFT, resource_name: <b>fft2</b>, ref_start_time: 5248428420073379, ref_stop_time: 5248428420160848, actual_exe_time: 87469
app_id: 0, app_name: track_nb, task_id: 14, task_name: DASH_FFT, resource_name: <b>fft2</b>, ref_start_time: 5248428420165289, ref_stop_time: 5248428420259618, actual_exe_time: 94329
app_id: 0, app_name: track_nb, task_id: 8, task_name: DASH_FFT, resource_name: <b>fft1</b>, ref_start_time: 5248428420242426, ref_stop_time: 5248428420336426, actual_exe_time: 94000
</pre>

We can see that all the resources available for FFT execution (`cpu1`, `cpu2`, `cpu3`, `fft1`, and `fft2`) are being used.

Lastly, let's compare the outputs with x86 results to validate the correctness.

```bash
xdg-open output_fft.png
cat output/pulse_doppler_output.txt
```

## 10. Contact

For any questions and bug report, please email to [suluhan@arizona.edu](mailto:suluhan@arizona.edu).
