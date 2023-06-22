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

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'start end sched_overhead proc')

def calc_idletime(proc_schedules):
    """
    Given a dictionary of processor-task schedules, calculates and prints PE idle time
    """  
    
    processors = list(proc_schedules.keys())


    ilen=len(processors)

    total_idle_time = 0
    queue_sched_overhead = 0
    total_queue_sched_overhead = 0
    for idx, proc in enumerate(processors):
        idle_time = 0
        queue_sched_overhead = 0
        #print("Length of Processor ", proc, " is ", len(proc_schedules[proc]), "\n")
        for i in range(len(proc_schedules[proc])-1):
            #print(proc_schedules[proc][i], "\n");
            idle_time += proc_schedules[proc][i+1].start - proc_schedules[proc][i].end
            queue_sched_overhead += proc_schedules[proc][i].sched_overhead
        # Adding the final entry scheduling overhead
        queue_sched_overhead += proc_schedules[proc][len(proc_schedules[proc])-1].sched_overhead
        print("Idle time for PE ",proc," is ", idle_time, " ns and Queued Scheduling overhead is ", queue_sched_overhead, " ns\n")
        total_idle_time += idle_time
        total_queue_sched_overhead += queue_sched_overhead
    print("Total idle time is ", total_idle_time, " ns and total Queued Scheduling overhead is ", total_queue_sched_overhead, " ns\n")



def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for calculating PE idle time and scheduling overhead from ZCU102 runtime")
    parser.add_argument("inputFile", help="Input trace file to be plotted as a Gantt chart")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    proc_schedules = {}

    with open(args.inputFile, newline='') as f:
       lines = f.readlines()
    lines = [(x.strip()).split(", ") for x in lines]
    start = sys.maxsize
    for elem in lines:
      start = min (start, int((elem[6].split(": "))[1]))
 

   # fields needed = processor, start_time, stop_time, scheduling overhead
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            proc_name = row[5].split(':')[1].strip()
            start_time = row[6].split(':')[1].strip()
            end_time = row[7].split(':')[1].strip()
            sched_overhead = row[9].split(':')[1].strip()
            schedule_event = ScheduleEvent((int(start_time) - start), (int(end_time)- start), (int(sched_overhead)), proc_name)
            if proc_name in proc_schedules:
                proc_schedules[proc_name].append(schedule_event)
            else:
                proc_schedules[proc_name] = [schedule_event]

    calc_idletime(proc_schedules)
