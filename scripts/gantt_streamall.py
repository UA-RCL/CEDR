"""
Basic implementation of Gantt chart plotting using Matplotlib
Taken from https://sukhbinder.wordpress.com/2016/05/10/quick-gantt-chart-with-matplotlib/ and adapted as necessary (i.e. removed Date logic, etc)

Intended for daemon-based execution with streaming enabled, and it plots all frames of a given application rather than just the first five frames
"""
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.patches import Patch
import numpy as np
import csv
import argparse
import sys
from collections import namedtuple
from colour import Color

ScheduleEvent = namedtuple('ScheduleEvent', 'app frame task start end proc')


def show_gantt_chart(processor_schedules, task_count, app_count, frame_count):
    """
    Given a dictionary of processor-task schedules, displays a Gantt chart generated using Matplotlib
    """

    processors = sorted(list(processor_schedules.keys()))

    # color_choices = ['maroon', 'brown', 'olive', 'teal', 'navy', 'black', 'red', 'orange', 'yellow',
    #                   'lime', 'green', 'cyan', 'blue', 'purple', 'magenta', 'gray', 'pink', 'apricot',
    #                   'beige', 'mint', 'lavender']
    color_choices = ['#800000', '#9A6324', '#808000', '#469990', '#000075', '#000000', '#E6194B', '#f58231', '#FFE119',
                     '#BFEF45', '#3CB44B', '#42D4F4', '#4363d8', '#911EB4', '#F032E6', '#A9A9A9', '#FABED4', '#FFD8B1',
                     '#FFFAC8', '#AAFFC3', '#DCBEFF']
    color_counts = len(color_choices)

    iLen = len(processors)
    pos = np.arange(0.5, iLen * 0.5 + 0.5, 0.5)
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    max_boxCount = 0
    if colorby == "tasks":
        for idx, proc in enumerate(processors):
            temp_box_count = 0
            for job in processor_schedules[proc]:
                ax.barh((idx * 0.5) + 0.5, (job.end - job.start) / 10 ** 6, left=job.start / 10 ** 6, height=0.3,
                        align='center',
                        edgecolor=color_choices[job.task % color_counts], color=color_choices[job.task % color_counts],
                        alpha=0.95)
                # ax.text(0.5 * (job.start + job.end - len(str(job.task))-0.25), (idx*0.5)+0.5 - 0.03125, job.task+1,
                # color=color_choices[job.job % 5], fontweight='bold', fontsize=18, alpha=0.75)
                temp_box_count += 1
            if temp_box_count > max_boxCount:
                max_boxCount = temp_box_count

    elif colorby == "apps":
        for idx, proc in enumerate(processors):
            temp_box_count = 0
            for job in processor_schedules[proc]:
                ax.barh((idx * 0.5) + 0.5, (job.end - job.start) / 10 ** 6, left=job.start / 10 ** 6, height=0.3,
                        align='center',
                        edgecolor=color_choices[job.app % color_counts], color=color_choices[job.app % color_counts],
                        alpha=0.95)
                temp_box_count += 1
            if temp_box_count > max_boxCount:
                max_boxCount = temp_box_count
    else:  # frames
        for idx, proc in enumerate(processors):
            temp_box_count = 0
            for job in processor_schedules[proc]:
                ax.barh((idx * 0.5) + 0.5, (job.end - job.start) / 10 ** 6, left=job.start / 10 ** 6, height=0.3,
                        align='center',
                        edgecolor=color_choices[job.frame % color_counts],
                        color=color_choices[job.frame % color_counts],
                        alpha=0.95)
                temp_box_count += 1
            if temp_box_count > max_boxCount:
                max_boxCount = temp_box_count

    locsy, labelsy = plt.yticks(pos, processors)
    plt.ylabel('Processor', fontsize=25)
    plt.xlabel('Time (ms)', fontsize=25)
    plt.setp(labelsy, fontsize=14)
    ax.set_ylim(ymin=-0.1, ymax=iLen * 0.5 + 0.8)
    if xMin != 0:
        ax.set_xlim(xmin=xMin)
    if xMax > 0:
        ax.set_xlim(xmax=xMax)
    ax.grid(color='g', linestyle=':', alpha=0.5)

    max_legend_count = legendCount  # default = 24; 6 columns, 4 rows
    legend_elements = []
    if colorby == "tasks":
        for i in range(min(task_count, max_legend_count)):
            legend_elements.append(
                Patch(facecolor=color_choices[i % color_counts], edgecolor=color_choices[i % color_counts],
                      label='TaskId ' + str(i)))
    elif colorby == "apps":
        for i in range(min(app_count, max_legend_count)):
            legend_elements.append(
                Patch(facecolor=color_choices[i % color_counts], edgecolor=color_choices[i % color_counts],
                      label='AppId ' + str(i)))
    else:  # frame
        for i in range(min(frame_count, max_legend_count)):
            legend_elements.append(
                Patch(facecolor=color_choices[i % color_counts], edgecolor=color_choices[i % color_counts],
                      label='FrameId ' + str(i)))

    ax.legend(handles=legend_elements, loc=9, ncol=7, fontsize=13)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=25)

    font = font_manager.FontProperties(size='small')
    if saveFig == None:
        plt.show()
    else:
        plt.savefig(saveFig)


def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for plotting Gantt charts from ZCU102 runtime",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputFile", help="Input trace file to be plotted as a Gantt chart")
    parser.add_argument("--color", nargs='?', const="tasks", type=str, default="tasks", choices=['tasks','apps','frames'],
                        help="Specify if the chart should be colored by tasks, apps or frames.")
    parser.add_argument("--xmin", nargs='?', const=0, type=int, default=0,
                        help="Set the minimum value along X-axis to start plotting from.")
    parser.add_argument("--xmax", nargs='?', const=0, type=int, default=0,
                        help="Set the maximum value along X-axis to stop plotting at.")
    parser.add_argument("--legendcount", nargs='?', const=21, type=int, default=21,
                        help="Set the maximum number of legends to display on plot.")
    parser.add_argument("--savefig", nargs='?', const=None, type=str, default=None,
                         help="Give filename (PNG) to save gantt chart as. Plot won't display if this option is used.")
    return parser


if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    proc_schedules = {}

    # Parse and store arguments in variables
    colorby = args.color
    xMin = args.xmin
    xMax = args.xmax
    legendCount = args.legendcount
    saveFig = args.savefig

    with open(args.inputFile, newline='') as f:
        lines = f.readlines()
    lines = [(x.strip()).split(", ") for x in lines]
    start = sys.maxsize
    for elem in lines:
        start = min(start, int((elem[6].split(": "))[1]))

    max_app_count = 0
    max_task_count = 0
    max_frame_count = 0
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            app_id = row[0].split(':')[1].strip()  # app_instances
            if max_app_count <= int(app_id):
                max_app_count = int(app_id) + 1
            frame_id = row[1].split(':')[1].strip()  # frame id
            if max_frame_count <= int(frame_id):
                max_frame_count = int(frame_id) + 1
            task_id = row[3].split(':')[1].strip()  # task id
            if max_task_count <= int(task_id):
                max_task_count = int(task_id) + 1
            proc_name = row[5].split(':')[1].strip()
            start_time = row[6].split(':')[1].strip()
            end_time = row[7].split(':')[1].strip()
            schedule_event = ScheduleEvent(int(app_id), int(frame_id), int(task_id), (int(start_time) - start),
                                           (int(end_time) - start), proc_name)
            if proc_name in proc_schedules:
                proc_schedules[proc_name].append(schedule_event)
            else:
                proc_schedules[proc_name] = [schedule_event]

    show_gantt_chart(proc_schedules, max_task_count, max_app_count, max_frame_count)
