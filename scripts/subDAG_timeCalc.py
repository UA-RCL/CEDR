"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)

Intended for daemon-based execution with streaming enabled, and it plots all frames of a given application rather than just the first five frames
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
import numpy as np
import glob, csv, argparse, sys

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'job task start end proc')

def create_gantt_chart(file, start, name):
    """
    Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """  
    proc_schedules = {}
    f = open(file, newline='')
    reader = csv.reader(f)
    for row in reader:
        job_id = row[0].split(':')[1].strip() # Use for coloring chart by application instances
        #job_id = row[1].split(':')[1].strip() # Use for coloring chart by frame number
        #if (int(job_id) < 5):  # Use to plot only first 5 frame/app instances
        task_id = row[3].split(':')[1].strip()
        proc_name = row[5].split(':')[1].strip()
        start_time = row[6].split(':')[1].strip()
        end_time = row[7].split(':')[1].strip()
        schedule_event = ScheduleEvent(int(job_id), int(task_id), (int(start_time) - start ), (int(end_time)- start), proc_name)
        if proc_name in proc_schedules:
            proc_schedules[proc_name].append(schedule_event)
        else:
            proc_schedules[proc_name] = [schedule_event]

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
    #ax.set_xlim(xmin = -5)     # Modify to set the minimum x limit of plotted chart
    #ax.set_xlim(xmax = 5)      # Modify to set the maximum x limit of plotted chart
    ax.grid(color = 'g', linestyle = ':', alpha=0.5)
    legend_elements = [Patch(facecolor=color_choices[0], edgecolor=color_choices[0],
                         label='Frame \n(0+5n)'),
                       Patch(facecolor=color_choices[1], edgecolor=color_choices[1],
                         label='Frame \n(1+5n)'),
                       Patch(facecolor=color_choices[2], edgecolor=color_choices[2],
                         label='Frame \n(2+5n)'),
                       Patch(facecolor=color_choices[3], edgecolor=color_choices[3],
                         label='Frame \n(3+5n)'),
                       Patch(facecolor=color_choices[4], edgecolor=color_choices[4],
                         label='Frame \n(4+5n)')]

    ax.legend(handles=legend_elements, loc=9, ncol=5,fontsize=16)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    #font = font_manager.FontProperties(size='small')
    plt.savefig(name + ".png")
    #plt.show()

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for calculating the average execution time of cedr instances")
    parser.add_argument("-p", "--path_to_experiments", help="Path folder where all the experiment folders contaninng the trace files")
    return parser

def find_start_time(file):
    f = open(file, newline='')
    lines = f.readlines()
    lines = [(x.strip()).split(", ") for x in lines]
    start = sys.maxsize
    start = int((lines[0][6].split(": "))[1])
    return start

def find_end_time(file):
    f = open(file, newline='')
    lines = f.readlines()
    lines = [(x.strip()).split(", ") for x in lines]
    end = 0
    for elem in lines:
        end = max (end, int((elem[7].split(": "))[1]))
    return end


def main():
    argparser = generate_argparser()
    args = argparser.parse_args()
    proc_schedules = {}
    files = glob.glob(args.path_to_experiments + "/exp*/timing_trace.log")
    print(files)
    startTimes = []
    endTime = []
    for file in files:
        startTimes += [find_start_time(file)]
        endTime += [find_end_time(file)]
    execTimes = [(endTime[i] - startTimes[i])/1e6 for i in range(len(startTimes))]


    print(execTimes)
    print("Average (ms): " + str(np.mean(execTimes)) #+ "\n" +
    #"Standard Deviation: " + str(np.std(execTimes) + "\n")
    )


    name = str(args.path_to_experiments.split("/")[-1])
    best = execTimes.index(min(execTimes))
    worst = execTimes.index(max(execTimes))
    create_gantt_chart(files[best], startTimes[best], name + "_best")
    create_gantt_chart(files[worst], startTimes[worst], name + "_worst")
    
if __name__ == "__main__":
    main()