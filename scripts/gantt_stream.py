"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)

Gantt script for plotting from daemon-based CEDR with streaming enabled, but it only plots the first 5 frames of the given application(s)
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
import numpy as np
import csv
import argparse
import sys

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'job task start end proc')

def show_gantt_chart(proc_schedules):
    """
    Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """  
    
    processors = list(proc_schedules.keys())

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
    color_choices = ['firebrick',  'midnightblue',  'lightskyblue', 'dodgerblue',  'green']
    ilen=len(processors)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:
            ax.barh((idx*0.5)+0.5, (job.end - job.start)/10**6, left=job.start/10**6, height=0.3, align='center', edgecolor=color_choices[job.job % 5], color=color_choices[job.job % 5], alpha=0.95)
            #ax.text(0.5 * (job.start + job.end - len(str(job.task))-0.25), (idx*0.5)+0.5 - 0.03125, job.task+1, color=color_choices[job.job % 5], fontweight='bold', fontsize=18, alpha=0.75)
    
    locsy, labelsy = plt.yticks(pos, processors)
    plt.ylabel('Processor', fontsize=25)
    plt.xlabel('Time (ms)', fontsize=25)
    plt.setp(labelsy, fontsize = 14)
    ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.8)
    #ax.set_xlim(xmin = -5)
    ax.grid(color = 'g', linestyle = ':', alpha=0.5)
    legend_elements = [Patch(facecolor=color_choices[0], edgecolor=color_choices[0],
                         label='Frame 0'),
                       Patch(facecolor=color_choices[1], edgecolor=color_choices[1],
                         label='Frame 1'),
                       Patch(facecolor=color_choices[2], edgecolor=color_choices[2],
                         label='Frame 2'),
                       Patch(facecolor=color_choices[3], edgecolor=color_choices[3],
                         label='Frame 3'),
                       Patch(facecolor=color_choices[4], edgecolor=color_choices[4],
                         label='Frame 4')]

    ax.legend(handles=legend_elements, loc=9, ncol=5,fontsize=16)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    #font = font_manager.FontProperties(size='small')
    plt.show()

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for plotting Gantt charts from ZCU102 runtime")
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
 

    
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            job_id = row[1].split(':')[1].strip()
            if (int(job_id) < 5):
             task_id = row[3].split(':')[1].strip()
             proc_name = row[5].split(':')[1].strip()
             start_time = row[6].split(':')[1].strip()
             end_time = row[7].split(':')[1].strip()
             schedule_event = ScheduleEvent(int(job_id), int(task_id), (int(start_time) - start ), (int(end_time)- start), proc_name)
             if proc_name in proc_schedules:
                 proc_schedules[proc_name].append(schedule_event)
             else:
                 proc_schedules[proc_name] = [schedule_event]

    show_gantt_chart(proc_schedules)
