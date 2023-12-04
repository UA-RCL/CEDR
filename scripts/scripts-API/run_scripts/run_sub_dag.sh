#!/bin/bash

#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "ETF")
declare -a CPUS=3
declare -a FFTS=0
declare -a MMULTS=0
declare -a ZIPS=0
declare -a GPUS=0
######################################################################

APPS=("./radar_correlator_fft-x86.so")
INSTS=("5")

declare -a PERIODS=("1734" "2313")
PERIODCOUNT=2

declare -a WORKLOADS=("HIGH")

FILE=launchfile

oldCedrPid="0"
cedrPid="0"
re="^[0-9]+$"

counter=0

for w in {0..0};do
  echo ${WORKLOADS[$w]} WORKLOADS
  echo "========================"
  for trial in {1..2}; do
    echo "Trial "${trial}
    echo "---------------"
    for ((period=0; period<PERIODCOUNT; period++)); do
      for (( gpu=0; gpu<=$GPUS; gpu++ )); do
        for (( zip=0; zip<=$ZIPS; zip++ )); do
          for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
            for (( fft=0; fft<=$FFTS; fft++ )); do
              for (( cpu=1; cpu<=$CPUS; cpu++ )); do
                for sched in "${SCHEDS[@]}"; do
                  while [ "$oldCedrPid" == "$cedrPid" ]
                  do # ! test -f $FILE ; do
                    #:
                    cedr=`ps aux | grep "cedr -c" | grep -v grep | tr -d '\n' | tr ' ' '\n'`
                    ID=`echo $cedr | tr ' ' '\n' | head -n 2 | tail -n 1` # Un-comment this for GPU
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
                  ./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"

                  sleep 2
                done
              done
            done
          done
        done
      done
    done
  done
done
