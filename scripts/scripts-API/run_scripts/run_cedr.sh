#!/bin/bash

# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
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
PERIODS=("1734" "2313" )

declare -a WORKLOADS=("HIGH" )

FILE=launchfile

counter=0
#trap "rm $FILE" EXIT

for w in {0..0};do
  for trial in {1..2}; do
    for ((period=0; period<PERIODCOUNT; period++)); do
      for (( gpu=0; gpu<=$GPUS; gpu++ )); do
        for (( zip=0; zip<=$ZIPS; zip++ )); do
          for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
            for (( fft=0; fft<=$FFTS; fft++ )); do
              for (( cpu=1; cpu<=$CPUS; cpu++ )); do
                for sched in "${SCHEDS[@]}"; do
                  echo "$counter": cpu ${cpu} fft ${fft} mmult ${mmult} zip ${zip} sched ${sched}
#                  echo "$counter": cpu ${cpu} fft ${fft} mmult ${mmult} zip ${zip}  gpu ${gpu} sched ${sched}
                  ./cedr -c schedsweep_daemon_configs/daemon_config-c${cpu}_f${fft}_m${mmult}_z${zip}-${sched}.json -l NONE > /dev/null 2>&1
#                  ./cedr -c schedsweep_daemon_configs/daemon_config-c${cpu}_f${fft}_m${mmult}_z${zip}_g${gpu}-${sched}.json -l NONE > /dev/null 2>&1
    	            #mv perf_stats.csv log_dir/experiment0/
                  # Rename log file
                  mv log_dir/experiment0 log_dir/c${cpu}_f${fft}_m${mmult}_z${zip}_sched-${sched}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
#                  mv log_dir/experiment0 log_dir/c${cpu}_f${fft}_m${mmult}_z${zip}_g${gpu}_sched-${sched}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
                  ((counter++))
                done
              done
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
