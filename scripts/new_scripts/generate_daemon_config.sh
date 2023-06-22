#!/bin/bash

if [[ $# -ne 5 ]]
then
        echo "There are $# argument(s)!"
        echo "Usage 'bash generate_daemon_config.sh CPU# FFT# GEMM# GPU#'"
        exit
fi

echo "Generating deamon config with $1 cpu(s), $2 fft(s), $3 gemm(s), $4 gpu(s), and $5 scheduler!"

echo "{
   \"Worker Threads\": {
        \"cpu\": $1,
        \"fft\": $2,
        \"gemm\": $3,
        \"gpu\": $4
    },

    \"Features\": {
        \"Cache Schedules\": false,
        \"Enable Queueing\": true,
        \"Use PAPI\": false,
        \"Loosen Thread Permissions\": true,
        \"Fixed Periodic Injection\": true,
        \"Exit When Idle\": true
    },

    \"PAPI Counters\": [
        \"perf::INSTRUCTIONS\",
        \"perf::BRANCHES\",
        \"perf::BRANCH-MISSES\",
        \"perf::L1-DCACHE-LOADS\",
        \"perf::L1-DCACHE-LOAD-MISSES\"
    ],

    \"DASH API Costs\": {
        \"DASH_FFT_cpu\": 20000,
        \"DASH_FFT_fft\": 21000,
        \"DASH_FFT_gpu\": 300,
        \"DASH_GEMM_cpu\": 15000,
        \"DASH_GEMM_mmult\": 500,
        \"DASH_GEMM_gpu\": 600
    },

    \"Scheduler\": \"$5\",
    \"Random Seed\": 0,
    \"DASH Binary Path\": [ \"./libdash-rt/libdash-rt.so\" ]
}" > daemon_config.json

