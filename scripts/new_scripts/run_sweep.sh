#!/bin/bash

# TODO: READ RESOURCE AND SCHEDULER LISTS, PERIODCOUNT AS INPUT ARGUMENT FROM A FILE
# Leave a copy of this input configuration file into the generated trace file base directory to be used
# later by makedataframe.
#### All possible schedulers, and max number of allowed resources ####
declare -a SCHEDS=("SIMPLE" "MET" "EFT" "ETF") # "HEFT_RT") 
CPUS=3
FFTS=1
MMULTS=1
ZIPS=0
GPUS=0
######################################################################

# Number of distinct period values for each workload. Use this value from the bash script that
# runs sub_dag
PERIODCOUNT=29
PERIODS=("101270,101270" "50635,50635" "33757,33757" "25317,25317" "20254,20254" "16878,16878" "14467,14467" "12659,12659" "11252,11252" "10127,10127" "5063,5063" "3376,3376" "2532,2532" "2025,2025" "1688,1688" "1447,1447" "1266,1266" "1125,1125" "1013,1013" "921,921" "844,844" "779,779" "723,723" "675,675" "633,633" "596,596" "563,563" "533,533" "506,506" "62500,62500" "2500,2500" "1250,1250" "833,833" "625,625" "500,500" "417,417" "357,357" "313,313" "278,278" "250,250" "227,227" "208,208" "192,192" "179,179" "167,167" "156,156" "147,147" "139,139" "125,125" "114,114" "104,104" "96,96" "89,89" "83,83" "78,78" "74,74" "69,69" "66,66")
#("0,1734,1734,2168" "0,2313,2313,2891" "0,3469,3469,4336" "0,6938,6938,8672" "0,34688,34688,43359" "0,346875,346875,433594" "0,1927,1927,2409" "0,2668,2668,3335" "0,4336,4336,5420" "0,11563,11563,14453" "0,69375,69375,86719" "0,2040,2040,2551" "0,2891,2891,3613" "0,4955,4955,6194" "0,17344,17344,21680" "0,49554,49554,61942" "0,115625,115625,144531" "0,43359,43359,54199" "0,173438,173438,216797" "0,1826,1826,2282" "0,2168,2168,2710" "0,3153,3153,3942" "0,5781,5781,7227" "0,38542,38542,48177" "0,86719,86719,108398" "0,2478,2478,3097" "0,3854,3854,4818" "0,8672,8672,10840" "0,57813,57813,72266")
APPS=("./wifi-tx-nb-aarch64.so,./pulse_doppler-nb-aarch64.so") # "apps/correlator.json,apps/temporal_mitigation.json")
INSTS=("5,5")

# Two different workloads, "High" has Pulse doppler and WiFi-TX; "Low" has Radar correlator and Temporal mitigation
declare -a WORKLOADS=("HIGH" ) #"LOW")

FILE=launchfile

counter=0
#trap "rm $FILE" EXIT

for w in {0..0};do
  for trial in {1..25}; do
    for ((period=0; period<$PERIODCOUNT; period++)); do
      for (( gpu=0; gpu<=$GPUS; gpu++ )); do
        for (( zip=0; zip<=$ZIPS; zip++ )); do
          for (( mmult=$MMULTS; mmult<=$MMULTS; mmult++ )); do
            for (( fft=$FFTS; fft<=$FFTS; fft++ )); do
              for (( cpu=$CPUS; cpu<=$CPUS; cpu++ )); do
                for sched in "${SCHEDS[@]}"; do
			bash generate_daemon_config.sh $cpu $fft $mmult $gpu $sched
			echo "$counter": cpu ${cpu} fft ${fft} mmult ${mmult} zip ${zip} sched ${sched} with ${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}
#                  echo "$counter": cpu ${cpu} fft ${fft} mmult ${mmult} zip ${zip}  gpu ${gpu} sched ${sched}
#                  ./cedr -c schedsweep_daemon_configs/daemon_config-c${cpu}_f${fft}_m${mmult}_z${zip}-${sched}.json -l NONE > /dev/null 2>&1
#                  ./cedr -c schedsweep_daemon_configs/daemon_config-c${cpu}_f${fft}_m${mmult}_z${zip}_g${gpu}-${sched}.json -l NONE > /dev/null 2>&1
			./cedr -c ./daemon_config.json -l NONE > /dev/null 2>&1 &
			sleep 5
			./sub_dag -a "${APPS[$w]}" -n "${INSTS[$w]}" -p "${PERIODS[(${w} * ${PERIODCOUNT}) + ${period}]}"
			sleep 5
			cedr_running=`ps aux | grep cedr | grep -v grep | wc -l`
			while [[ $cedr_running -ne 0 ]]
			do
				sleep 2
				cedr_running=`ps aux | grep cedr | grep -v grep | wc -l`
			done
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
