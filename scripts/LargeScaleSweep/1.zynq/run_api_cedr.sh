#!/bin/bash

# TODO: READ RESOURCE AND SCHEDULER LISTS, PERIODCOUNT AS INPUT ARGUMENT FROM A FILE
# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "EFT" "ETF") # "HEFT_RT") 
CPUS=3
FFTS=8
MMULTS=2
ZIPS=2
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag
PERIODCOUNT=29
PERIODS=("0,346875,346875,433594" "0,173438,173438,216797" "0,115625,115625,144531" "0,86719,86719,108398" "0,69375,69375,86719" "0,57813,57813,72266" "0,49554,49554,61942" "0,43359,43359,54199" "0,38542,38542,48177" "0,34688,34688,43359" "0,17344,17344,21680" "0,11563,11563,14453" "0,8672,8672,10840" "0,6938,6938,8672" "0,5781,5781,7227" "0,4955,4955,6194" "0,4336,4336,5420" "0,3854,3854,4818" "0,3469,3469,4336" "0,3153,3153,3942" "0,2891,2891,3613" "0,2668,2668,3335" "0,2478,2478,3097" "0,2313,2313,2891" "0,2168,2168,2710" "0,2040,2040,2551" "0,1927,1927,2409" "0,1826,1826,2282" "0,1734,1734,2168")

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
declare -a WORKLOADS=("HIGH" ) #"LOW")

FILE=launchfile

counter=0
#trap "rm $FILE" EXIT

for w in {0..0};do
  for trial in {1..1}; do
    for ((period=0; period<PERIODCOUNT; period++)); do
      for (( zip=0; zip<=$ZIPS; zip++ )); do
        for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
          for (( fft=0; fft<=$FFTS; fft++ )); do
            for (( cpu=1; cpu<=$CPUS; cpu++ )); do
              for sched in "${SCHEDS[@]}"; do
                echo "$counter": cpu ${cpu} fft ${fft} mmult ${mmult} zip ${zip} sched ${sched}
                ./cedr -c schedsweep_daemon_configs/daemon_config-c${cpu}_f${fft}_m${mmult}_z${zip}-${sched}.json -l NONE > /dev/null 2>&1
  	            mv perf_stats.csv log_dir/experiment0/
                # Rename log file
                mv log_dir/experiment0 log_dir/c${cpu}_f${fft}_m${mmult}_z${zip}_sched-${sched}-p"${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
                #echo cpu ${cpu} fft ${fft} mmult ${mmult} sched ${sched}
                ((counter++))
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
