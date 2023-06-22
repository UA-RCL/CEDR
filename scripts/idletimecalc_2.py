"""
Intended for parsing the timing trace files from daemon-based execution with streaming enabled, and calculating the following quantities-
1. Resource idle time (white-space) per resource
2. Scheduling overhead for thread level queues enabled- per resource
3. Total Resource idle time and Scheduling overhead over all the resources.

Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)

"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
import numpy as np
import csv
import argparse
import sys

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for calculating PE idle time and scheduling overhead from ZCU102 runtime")
    parser.add_argument("inputFile", help="Input trace file to be plotted as a Gantt chart")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    with open(args.inputFile, newline='') as f:
       lines = f.readlines()
    lines = [(x.strip()).split(", ") for x in lines]
    start = sys.maxsize
    end = 0
    for elem in lines:
      start = min (start, int((elem[6].split(": "))[1]))
      end = max (end, int((elem[7].split(": "))[1]))
    
   # fields needed = processor, start_time, stop_time, scheduling overhead
    PE_starts = {}
    PE_ends = {}
    PE_exec_times = {}
    Total_time = 0
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            PE = row[5].split(':')[1].strip()
            start_time = int(row[6].split(':')[1].strip())
            end_time = int(row[7].split(':')[1].strip())
            if PE in PE_starts:
                if start_time < PE_starts[PE]:
                    PE_starts[PE] = start_time
            else:
                PE_starts[PE] = start_time
            if PE in PE_ends:
                if end_time > PE_ends[PE]:
                    PE_ends[PE] = end_time
            else:
                PE_ends[PE] = end_time
            if PE in PE_exec_times:
                PE_exec_times[PE] += end_time - start_time
            else:
                PE_exec_times[PE] = end_time - start_time
    for PE in PE_starts.keys():
        print("Execution time  of PE", PE, " is:", PE_exec_times[PE]/1e6)
        print("Execution time (inclduing Idle) of PE :", PE, " is:", ((PE_ends[PE] - PE_starts[PE]))/1e6)
        print("Idle time from PE perspective:", ((PE_ends[PE] - PE_starts[PE])- PE_exec_times[PE])/1e6)
        print("Idle time from beginning of sub_dag", ((PE_ends[PE] - start)- PE_exec_times[PE])/1e6, "\n")
        Total_time += ((PE_ends[PE] - start)- PE_exec_times[PE])/1e6
    print("Total time(ms): ", Total_time)           
