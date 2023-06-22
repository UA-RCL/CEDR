"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)

Baseline implementation for use with non-daemon CEDR, non-streaming execution
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import csv
import argparse

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'job task start end proc')

def show_gantt_chart(proc_schedules, app_names):
    """
    Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """  
    
    processors = sorted(list(proc_schedules.keys()))

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']

    ilen=len(processors)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:
            ax.barh((idx*0.5)+0.5, (job.end - job.start)/10**6, left=job.start/10**6, height=0.3, align='center', edgecolor=color_choices[job.job % 5], color=color_choices[job.job % 5], alpha=0.95)
            #ax.text(0.5 * (job.start + job.end - len(str(job.task))-0.25), (idx*0.5)+0.5 - 0.03125, job.task+1, color=color_choices[job.job % 5], fontweight='bold', fontsize=18, alpha=0.75)
    
    locsy, labelsy = plt.yticks(pos, processors)
    plt.ylabel('Processor', fontsize=16)
    plt.xlabel('Time (ms)', fontsize=16)
    plt.setp(labelsy, fontsize = 14)
    ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.5)
    ax.set_xlim(xmin = -5)
    ax.grid(color = 'g', linestyle = ':', alpha=0.5)
    plt.legend(app_names.keys())
    font = font_manager.FontProperties(size='small')
    plt.show()

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for plotting Gantt charts from ZCU102 runtime")
    parser.add_argument("inputFile", help="Input trace file to be plotted as a Gantt chart")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    proc_schedules = {}
    applications_seen = {}
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            job_id = row[0].split(':')[1].strip()
            app_name = row[1].split(':')[1].strip()
            task_id = row[2].split(':')[1].strip()
            proc_name = row[4].split(':')[1].strip()
            start_time = row[5].split(':')[1].strip()
            end_time = row[6].split(':')[1].strip()
            schedule_event = ScheduleEvent(int(job_id), int(task_id), int(start_time), int(end_time), proc_name)
            if proc_name in proc_schedules:
                proc_schedules[proc_name].append(schedule_event)
            else:
                proc_schedules[proc_name] = [schedule_event]
            if app_name not in applications_seen:
                applications_seen[app_name] = job_id

    print(applications_seen) 
    show_gantt_chart(proc_schedules, applications_seen)
