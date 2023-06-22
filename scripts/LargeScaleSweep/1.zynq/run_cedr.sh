#!/bin/bash

# TODO: READ RESOURCE AND SCHEDULER LISTS, PERIODCOUNT AS INPUT ARGUMENT FROM A FILE
# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "EFT" "ETF" "HEFT_RT")
CPUS=3
FFTS=1
MMULTS=1
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag
PERIODCOUNT=29

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
declare -a WORKLOADS=("HIGH" "LOW")

FILE=launchfile

#trap "rm $FILE" EXIT

for w in {0..1};do
  for trial in {1..5}; do
    for ((period=0; period<PERIODCOUNT; period++)); do
      for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
        for (( fft=0; fft<=$FFTS; fft++ )); do
          for (( cpu=1; cpu<=$CPUS; cpu++ )); do
            for sched in "${SCHEDS[@]}"; do
              sleep 2
              touch $FILE
              ./cedr -c schedsweep_daemon_configs/daemon_config-c${cpu}_f${fft}_m${mmult}-${sched}.json -l NONE > /dev/null 2>&1
	      mv perf_stats.csv log_dir/experiment0/
              # Rename log file
              mv log_dir/experiment0 log_dir/c${cpu}_f${fft}_m${mmult}_sched-${sched}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
              #echo cpu ${cpu} fft ${fft} mmult ${mmult} sched ${sched}
            done
          done
        done
      done
    done
    mkdir log_dir/trial_${trial}
    mv log_dir/c* log_dir/trial_${trial}
  done
  mkdir log_dir/${WORKLOADS[$w]}
  mv log_dir/trial_* log_dir/${WORKLOADS[$w]}
done
