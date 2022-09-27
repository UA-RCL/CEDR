#!/usr/bin/python3

import argparse
import json

baseline_config = {
    "Worker Threads": {
        "cpu": 3,
        "fft": 0,
        "mmult": 0,
        "gpu": 0
    },

    "Features": {
        "Cache Schedules": False,
        "Enable Queueing": True,
        "Use PAPI": False,
        "Loosen Thread Permissions": False,
        "Fixed Periodic Injection": True,
        "Exit When Idle": False
    },

    "PAPI Counters": [
        "perf::INSTRUCTIONS",
        "perf::BRANCHES",
        "perf::BRANCH-MISSES",
        "perf::L1-DCACHE-LOADS",
        "perf::L1-DCACHE-LOAD-MISSES"
    ],

    "Scheduler": "SIMPLE",
    "Random Seed": 0
}

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for generating a config file for use with the CEDR daemon")

    parser.add_argument("-s", "--scheduler", help="Choose the scheduler to use in CEDR", type=str, default="SIMPLE")

    parser.add_argument("--cache-schedules", help="Once a node is scheduled, future iterations of that node will use the same decision", type=str2bool, default=False)
    parser.add_argument("-q", "--queue_enable", help="Bool flag to enable/disable thread-level queueing", type=str2bool, default=True)
    parser.add_argument("-p", "--papi_enable", help="Bool flag to enable/disable PAPI counters", type=str2bool, default=False)
    parser.add_argument("-l", "--loosen_thread_permissions", help="Bool flag to enable/disable thread permissions", type=str2bool, default=False)
    parser.add_argument("-f", "--fixed_periodic_injection", help="Bool flag to enable/disable fixed periodic injection of instances", type=str2bool, default=True)
    parser.add_argument("-e", "--exit_when_idle", help="Bool flag to enable/disable CEDR to exit after being done with 1st sub_dag", type=str2bool, default=False)

    parser.add_argument("-o", "--output-file", help="Name of the generated output JSON file", type=str, default="daemon_config.json")
    parser.add_argument("-cpus", "--CPUS", help="CPUS in the generated JSON file", type=int, default=3)
    parser.add_argument("-ffts", "--FFTS", help="FFTS in the generated JSON file", type=int, default=0)
    parser.add_argument("-mmults", "--MMULTS", help="MMULTS in the generated JSON file", type=int, default=0)
    parser.add_argument("-gpus", "--GPUS", help="GPUS in the generated JSON file", type=int, default=0)

    parser.add_argument("-il", "--IL_COUNTERS", help="Have PAPI Counters for IL training", type=str2bool, default=False)
    return parser

'''
Proper string to bool conversation. Python parsers take any string as True if you say
the desired type is a bool.
'''
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): #Already covers TRUE, True, etc
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): #Already covers FALSE, False, etc
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()
    baseline_config["Features"]["Cache Schedules"] = args.cache_schedules
    baseline_config["Features"]["Enable Queueing"] = args.queue_enable
    baseline_config["Features"]["Use PAPI"] = args.papi_enable
    baseline_config["Features"]["Loosen Thread Permissions"] = args.loosen_thread_permissions
    baseline_config["Features"]["Fixed Periodic Injection"] = args.fixed_periodic_injection
    baseline_config["Features"]["Exit When Idle"] = args.exit_when_idle
    baseline_config["Scheduler"] = args.scheduler
    baseline_config["Worker Threads"]["cpu"] = args.CPUS
    baseline_config["Worker Threads"]["fft"] = args.FFTS
    baseline_config["Worker Threads"]["mmult"] = args.MMULTS
    baseline_config["Worker Threads"]["gpu"] = args.GPUS
    if(args.IL_COUNTERS):
      baseline_config["PAPI Counters"] = [  
        "perf::LLC-LOAD-MISSES",
        "INST_RETIRED",
        "CPU_CYCLES",
        "BRANCH_MISPRED",
        "DATA_MEM_ACCESS"
      ]
       

    with open(args.output_file, 'w') as fp:
        json.dump(baseline_config, fp, indent=4)