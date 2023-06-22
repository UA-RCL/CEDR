#!/bin/bash

# TODO: READ SCHEDULERS, RESOURCES PERIODCOUNT AND PERIODS AS INPUT ARGUMENTS
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "EFT" "ETF" "HEFT_RT") 
declare -a CPUS=3
declare -a FFTS=1
declare -a MMULTS=1
######################################################################

#### Two sets of workload and their corresponding number of instances #################
#### , High => pulse doppler, wifitx; Low => radar correlator, temporal mitigation ####
DAGS=("apps/TX-aarch64.json,apps/pulse_doppler-aarch64.json" "apps/correlator.json,apps/temporal_mitigation.json")
INSTS=("5,5" "20,20")


# This array contains 58 injection rates, PERIODS[0:28] corresponds to "High" workload injection rates, and PERIODS[29:57] corresponds to "Low" workload injection rates.
declare -a PERIODS=("101270,101270" "50635,50635" "33757,33757" "25317,25317" "20254,20254" "16878,16878" "14467,14467" "12659,12659" "11252,11252" "10127,10127" "5063,5063" "3376,3376" "2532,2532" "2025,2025" "1688,1688" "1447,1447" "1266,1266" "1125,1125" "1013,1013" "921,921" "844,844" "779,779" "723,723" "675,675" "633,633" "596,596" "563,563" "533,533" "506,506" "62500,62500" "2500,2500" "1250,1250" "833,833" "625,625" "500,500" "417,417" "357,357" "313,313" "278,278" "250,250" "227,227" "208,208" "192,192" "179,179" "167,167" "156,156" "147,147" "139,139" "125,125" "114,114" "104,104" "96,96" "89,89" "83,83" "78,78" "74,74" "69,69" "66,66")

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
PERIODCOUNT=29

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
declare -a WORKLOADS=("HIGH" "LOW")

FILE=launchfile

for w in {0..1};do
  echo ${WORKLOADS[$w]} WORKLOADS
  echo "========================"
  for trial in {1..5}; do
    echo "Trial "${trial}
    echo "---------------"
    for ((period=0; period<PERIODCOUNT; period++)); do
      for (( mmult=0; mmult<=$MMULTS; mmult++ )); do
        for (( fft=0; fft<=$FFTS; fft++ )); do
          for (( cpu=1; cpu<=$CPUS; cpu++ )); do
            for sched in "${SCHEDS[@]}"; do
              while ! test -f $FILE ; do
                :
              done
	      sleep 2
	      echo ./sub_dag -d "${DAGS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
	      ./sub_dag -d "${DAGS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
              rm $FILE
              # If needed to call out which experiment is running, uncomment following line
	      #echo cpu ${cpu} fft ${fft} mmult ${mmult} sched ${sched}
            done
          done
        done
      done
    done
  done
done
