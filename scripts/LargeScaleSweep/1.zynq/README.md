# Scripts for running large scale CEDR experiments on ZCU102

## File descriptions
1. `run_cedr.sh`: This script launches CEDR with proper configuration (Resource count and types, scheduler etc.) over multiple iterations of the large scale experiment. It also organizes the generated trace files by their respective input configurations in different directories. 
2. `run_sub_dag.sh`: This script calls `sub_dag` with proper application dag, number of instances and periodicity arguments.
3. `schedsweep_daemon_configs/`: This folder contains all daemon configurations for running large scale experiment.
4. `daemon_generator.py`: This script can be used to generate desired daemon configuration files. Also, if any changes are needed to be made in current set of daemon configuration files (in `schedsweep_daemon_configs/`folder), this script can be modified to produce new files with those changes.


## How to run the experiments on ZCU102
1. Copy `run_cedr.sh`, `run_sub_dag.sh`, `schedsweep_daemon_configs/` on zynq, inside the folder that contains the `cedr`, `sub_dag` and `kill_daemon` binaries. This folder should also contain the desired application DAGs and shared objects inside `apps/` folder. 

* ***N.B.: The variables `PERIODCOUNT`, `PERIODS` and `WORKLOADS` array in both `run_cedr.sh` and `run_sub_dag.sh` should be consistent.*** 

2. Open two terminals over ssh. First, in terminal 1 launch the `run_cedr.sh` script. Then, in terminal 2 launch `run_sub_dag.sh` bash script.
3. Terminal 2 should keep printing as it progresses through multiple trials of each experiment. The description of experiments conducted so far with these scripts can be found in the [CEDR paper (TECS)](https://arxiv.org/abs/2204.08962), Section 3, table 3.

## Structure of generated output files
The large scale experiment produces three kinds of output files for each CEDR configuration- timing trace file, scheduler overhead trace file, and performance counter output file. These files are generated inside the `log_dir` folder.
The structure of files for different trials and input configurations are organized in the following manner

```bash
|____log_dir
| |____HIGH
| | |____trial 1
| | | |____c<CPU>_f<FFT>_m<MMULT>_sched-<SCHEDULER>-p<PERIOD>/
| | | | |____timing_trace.log
| | | | |____schedule_trace.log
| | | | |____perf_stats.csv
| | | |____....
| | | |____....
| | | |____....
| | | |____....
| | |____trial 2
| | | |____c<CPU>_f<FFT>_m<MMULT>_sched-<SCHEDULER>-p<PERIOD>/
| | | | |____timing_trace.log
| | | | |____schedule_trace.log
| | | | |____perf_stats.csv
| | | |____....
| | | |____....
| | | |____....
| | | |____....
| | |____........
| | |____trial N
| |____LOW
| | |____trial 1
| | | |____c<CPU>_f<FFT>_m<MMULT>_sched-<SCHEDULER>-p<PERIOD>/
| | | | |____timing_trace.log
| | | | |____schedule_trace.log
| | | | |____perf_stats.csv
| | | |____....
| | | |____....
| | | |____....
| | | |____....
| | |____........
| | |____trial N
```

This generated directory will be used by scripts inside `2.trace_extract/` to extract all the data into a single CSV file, that can then be used for plotting and analysis. 