#!/bin/bash

# TODO: READ SCHEDULERS, RESOURCES PERIODCOUNT AND PERIODS AS INPUT ARGUMENTS
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "EFT" "ETF") # "HEFT_RT") 
declare -a CPUS=3
declare -a FFTS=8
declare -a MMULTS=2
declare -a ZIPS=2
######################################################################

#### Two sets of workload and their corresponding number of instances #################
#### , High => pulse doppler, wifitx; Low => radar correlator, temporal mitigation ####
APPS=("apps/track_fft.so,apps/TX-aarch64.so,apps/pulse_doppler-aarch64.so") # "apps/correlator.json,apps/temporal_mitigation.json")
INSTS=("1,25,25,20")


# This array contains 58 injection rates, PERIODS[0:28] corresponds to "High" workload injection rates, and PERIODS[29:57] corresponds to "Low" workload injection rates.
declare -a PERIODS=("0,346875,346875,433594" "0,173438,173438,216797" "0,115625,115625,144531" "0,86719,86719,108398" "0,69375,69375,86719" "0,57813,57813,72266" "0,49554,49554,61942" "0,43359,43359,54199" "0,38542,38542,48177" "0,34688,34688,43359" "0,17344,17344,21680" "0,11563,11563,14453" "0,8672,8672,10840" "0,6938,6938,8672" "0,5781,5781,7227" "0,4955,4955,6194" "0,4336,4336,5420" "0,3854,3854,4818" "0,3469,3469,4336" "0,3153,3153,3942" "0,2891,2891,3613" "0,2668,2668,3335" "0,2478,2478,3097" "0,2313,2313,2891" "0,2168,2168,2710" "0,2040,2040,2551" "0,1927,1927,2409" "0,1826,1826,2282" "0,1734,1734,2168")

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
PERIODCOUNT=29

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
declare -a WORKLOADS=("HIGH") # "LOW")

FILE=launchfile

oldCedrPid="0"
cedrPid="0"
re="^[0-9]+$"

counter=0

for w in {0..0};do
  echo ${WORKLOADS[$w]} WORKLOADS
  echo "========================"
  for trial in {1..1}; do
    echo "Trial "${trial}
    echo "---------------"
    for ((period=0; period<PERIODCOUNT; period++)); do
      for (( zip=0; zip<=$ZIPS; zip++ )); do
        for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
          for (( fft=0; fft<=$FFTS; fft++ )); do
            for (( cpu=1; cpu<=$CPUS; cpu++ )); do
              for sched in "${SCHEDS[@]}"; do
                while [ "$oldCedrPid" == "$cedrPid" ]
                do # ! test -f $FILE ; do
                  #:
                  cedr=`ps aux | grep "cedr -c" | grep -v grep | tr -d '\n' | tr ' ' '\n'`
                  ID=`echo $cedr | tr ' ' '\n' | head -n 1` #2 | tail -n 1`
                  time=`echo $cedr | tr ' ' '\n' | head -n 3 | tail -n 1 | tr ':' '\n' | head -n 1`
                  if [[ "$ID" =~ ^[0-9]+$ ]]
                  then
                    cedrPid=$ID
                  fi
                  sleep 2
                  #echo $ID
                  #echo $cedrPid
                  #exit
                done
                oldCedrPid=$cedrPid
                sleep 1
                echo "$counter": ./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
                ((counter++))
                ./sub_dag -d "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"

                sleep 2
              done
            done
          done
        done
      done
    done
  done
done
