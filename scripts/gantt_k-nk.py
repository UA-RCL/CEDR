"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)

Intended for daemon-based execution with streaming enabled, and it plots all frames of a given application rather than just the first five frames
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch, Rectangle
import numpy as np
import csv
import argparse
import sys

from collections import namedtuple

ScheduleEvent = namedtuple('ScheduleEvent', 'job task start end proc id_string')

def show_gantt_chart(proc_schedules):
    """
    Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """  
    
    processors = sorted(list(proc_schedules.keys()))

    color_choices = ['red', 'blue', 'green', 'cyan', 'magenta']
    color_choices = ['firebrick',  'midnightblue',  'lightskyblue', 'dodgerblue',  'green']
    ilen=len(processors)
    pos = np.arange(0.5,ilen*0.5+0.5,0.5)
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)
    rect_containers = []
    toolbar_width = 50
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    progressbar = 0
    for idx, proc in enumerate(processors):
        progressbar=progressbar+len(proc_schedules[proc])
    percantage=progressbar
    percantage_counter=0
    progressbar=(progressbar+1)/toolbar_width - 1
    progressbar_counter=0
    for idx, proc in enumerate(processors):
        for job in proc_schedules[proc]:
            progressbar_counter = progressbar_counter + 1
            rect_containers.append([ax.barh((idx*0.5)+0.5, (job.end - job.start)/10**6, left=job.start/10**6, height=0.3, align='center', edgecolor=color_choices[job.task % 5], color=color_choices[job.task % 5], alpha=0.95), job.id_string])
            #ax.text(0.5 * (job.start + job.end - len(str(job.task))-0.25), (idx*0.5)+0.5 - 0.03125, job.task+1, color=color_choices[job.job % 5], fontweight='bold', fontsize=18, alpha=0.75)
            if progressbar_counter >= progressbar:
                percantage_counter+=progressbar_counter
                sys.stdout.write("\r")
                sys.stdout.write("[%s" % ("=" * int(percantage_counter/progressbar_counter)))
                sys.stdout.write("%s]" % (" " * (toolbar_width - int(percantage_counter/progressbar_counter))))
                sys.stdout.write("{:.2f}".format(percantage_counter/percantage*100.0))
                sys.stdout.write("%")
                sys.stdout.flush()
                progressbar_counter = 0
    sys.stdout.write("\r")
    sys.stdout.write("[%s]100" % ("=" * toolbar_width))
    sys.stdout.write("%")
    sys.stdout.write("  \n")
    sys.stdout.flush()

    if annotation_enable == True:
        # Initialize text annotation 
        annot = ax.annotate("", xy = (0,0), xytext = (5,5), textcoords = "offset points")
        annot.set_visible(False)

    # Add labels, legends and formatting of graph
    locsy, labelsy = plt.yticks(pos, processors)
    plt.ylabel('Processor', fontsize=25)
    plt.xlabel('Time (ms)', fontsize=25)
    plt.setp(labelsy, fontsize = 14)
    ax.set_ylim(ymin = -0.1, ymax = ilen*0.5+0.8)
    #ax.set_xlim(xmin = -5)     # Modify to set the minimum x limit of plotted chart
    #ax.set_xlim(xmax = 5)      # Modify to set the maximum x limit of plotted chart
    ax.grid(color = 'g', linestyle = ':', alpha=0.5)
    legend_elements = [Patch(facecolor=color_choices[0], edgecolor=color_choices[0],
                         label='TASK_ID \n(0+5n)'),
                       Patch(facecolor=color_choices[1], edgecolor=color_choices[1],
                         label='TASK_ID \n(1+5n)'),
                       Patch(facecolor=color_choices[2], edgecolor=color_choices[2],
                         label='TASK_ID \n(2+5n)'),
                       Patch(facecolor=color_choices[3], edgecolor=color_choices[3],
                         label='TASK_ID \n(3+5n)'),
                       Patch(facecolor=color_choices[4], edgecolor=color_choices[4],
                         label='TASK_ID \n(4+5n)')]

    ax.legend(handles=legend_elements, loc=9, ncol=5,fontsize=16)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)


    if annotation_enable == True:
        # Define function to be used at event of hover/click
        def hover (event):
            if event.inaxes == ax:
                # Check if the event overlaps on one of the rectangles
                for rect in rect_containers:
                    rectangle = rect[0][0]
                    string_id = rect[1]
                    bbox = rectangle.get_bbox()
                    if bbox.fully_contains(event.xdata, event.ydata):
                        annot.xy = (event.xdata, event.ydata)
                        annot.set_text(string_id)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        break
                    else:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
            else:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        # Enable hover event tracking
        fig.canvas.mpl_connect("motion_notify_event", hover)
    
    #font = font_manager.FontProperties(size='small')
    plt.show()
    plt.savefig("gantt.png")

def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for plotting Gantt charts from ZCU102 runtime")
    parser.add_argument("inputFile", help="Input trace file to be plotted as a Gantt chart")
    parser.add_argument('--no-annot', dest='annot_enable', action='store_true', 
        help="Enable annotation of task rectangle while hovered upon.")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    annotation_enable = True
    if args.annot_enable == True:
        annotation_enable = False

    proc_schedules = {}

    with open(args.inputFile, newline='', encoding='utf-8') as f:
       lines = f.readlines()
    lines = [(x.strip()).split(", ") for x in lines]
    start = sys.maxsize
    for elem in lines:
      start = min (start, int((elem[5].split(": "))[1]))
 

    
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            job_id = row[0].split(':')[1].strip() # Use for coloring chart by application instances
            #if (int(job_id) < 5):  # Use to plot only first 5 frame/app instances
            job_name = row[1].split(':')[1].strip()
            task_id = row[2].split(':')[1].strip()
            task_name = row[3].split(':')[1].strip()
            proc_name = row[4].split(':')[1].strip()
            start_time = row[5].split(':')[1].strip()
            end_time = row[6].split(':')[1].strip()
            task_identifier = job_name + '_' + job_id + '-' + task_name
            schedule_event = ScheduleEvent(int(job_id), int(task_id), (int(start_time) - start ), (int(end_time)- start), proc_name, task_identifier)
            if proc_name in proc_schedules:
                proc_schedules[proc_name].append(schedule_event)
            else:
                proc_schedules[proc_name] = [schedule_event]

    show_gantt_chart(proc_schedules)
