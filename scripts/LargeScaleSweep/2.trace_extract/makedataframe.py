import numpy as np
import pandas as pd
#import plotly.express as px
import csv
from collections import Counter, namedtuple
import argparse

"""
TODO: 
1. Add `median` option too along with `min` and `avg` cases.
2. Use the experiment configuration file left out in the trace base directory to configure
the input parameters such as, resource names and counts, scheduler names, injection rates. 
"""

ScheduleEvent = namedtuple("ScheduleEvent", 'resource start stop exectime')

def resource_util(FILENAME):

    proc_schedules = {}
    with open(FILENAME, newline='') as fp:
        reader = csv.reader(fp)
        for row in reader:
            resource_name = row[5].split(':')[1]
            start_time = row[6].split(':')[1]
            stop_time = row[7].split(':')[1]
            execution_time = row[8].split(':')[1]

            schedule_event = ScheduleEvent(resource_name, (int(start_time)), (int(stop_time)), (int(execution_time)))

            if resource_name in proc_schedules:
                proc_schedules[resource_name].append(schedule_event)
            else:
                proc_schedules[resource_name] = [schedule_event]

        start_dict = {}
        stop_dict = {}
        exectime_dict = {}
        corelist = [' Core 1', ' Core 2', ' Core 3', ' FFT 1', ' MMULT 1'] # Edit here
        #print("Proc_schedule_keys : ", proc_schedules.keys())
        for core in corelist:
            if core in proc_schedules.keys():
                for sched in proc_schedules[core]:
                    if core in start_dict.keys():
                        start_dict[str(core)].append(sched[1])
                        stop_dict[str(core)].append(sched[2])
                        exectime_dict[str(core)].append(sched[3])
                    else:
                        start_dict[str(core)] = [sched[1]]
                        stop_dict[str(core)] = [sched[2]]
                        exectime_dict[str(core)] = [sched[3]]
            else:
                exectime_dict[str(core)] = [0]

        coreutil_dict = {}
        for core in corelist:
            if core in start_dict.keys():
                start_dict[str(core)] = min(start_dict[str(core)])
                stop_dict[core] = max(stop_dict[core])
            exectime_dict[core] = sum(exectime_dict[core])
        start_minimum = min(start_dict.values())
        stop_maximum = max(stop_dict.values())

        for core in corelist:
            coreutil_dict[core] = exectime_dict[core] / (stop_maximum - start_minimum)

        return coreutil_dict

# Add dictionaries with zero elements too
def add_dicts(a, b):
    return {x: a.get(x,0) + b.get(x,0) for x in set(a).union(b)}


def schedule_overhead_total(FILENAME):
    with open(FILENAME, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if 'total_scheduling_overhead' in row[1]:
                task_count = int(row[0].split(':')[1].strip())
                scheduling_overhead_total = int(row[1].split(':')[1].strip().replace(' ns',''))
    #print ("schedling overhead in total is ", scheduling_overhead_total)
    #print("Task count ", task_count)
    return scheduling_overhead_total, task_count

def cumulative_exec(FILENAME):

    app_starts = {}
    app_ends = {}

    # Determine the start and end time of each application
    with open(FILENAME, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            app_id = int(row[0].split(':')[1].strip())
            start_time = int(row[6].split(':')[1].strip())
            end_time = int(row[7].split(':')[1].strip())

            if app_id in app_starts:
                if start_time < app_starts[app_id]:
                    app_starts[app_id] = start_time
            else:
                app_starts[app_id] = start_time

            if app_id in app_ends:
                if end_time > app_ends[app_id]:
                    app_ends[app_id] = end_time
            else:
                app_ends[app_id] = end_time

    # Determine the cumulative execution time and "average job execution time" a la DS3
    cumulative_exec = 0 
    for app_idx in app_starts.keys():
        cumulative_exec += app_ends[app_idx] - app_starts[app_idx]

    num_jobs = max(app_starts.keys()) + 1 # Offset the 0-based app ids

    return cumulative_exec/num_jobs, cumulative_exec, num_jobs 
    # print(f"The cumulative execution time was {cumulative_exec} over {num_jobs} jobs, giving an average execution of {cumulative_exec / num_jobs}")

def cumulative_exec_only(FILENAME):

    app_exectime = {}

    # Determine the start and end time of each application
    with open(FILENAME, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            app_id = int(row[0].split(':')[1].strip())
            start_time = int(row[6].split(':')[1].strip())
            end_time = int(row[7].split(':')[1].strip())

            if app_id in app_exectime:
                app_exectime[app_id] += (end_time - start_time)
            else:
                app_exectime[app_id] = (end_time - start_time)

    # Determine the cumulative execution time and "average job execution time" a la DS3
    cumulative_exec = 0 
    for app_idx in app_exectime.keys():
        cumulative_exec += app_exectime[app_idx]

    num_jobs = max(app_exectime.keys()) + 1 # Offset the 0-based app ids

    return cumulative_exec/num_jobs, cumulative_exec, num_jobs 

def generate_argparser():
    parser = argparse.ArgumentParser(description='Python script to dump timing trace stats into a csv file!')
    parser.add_argument('-i', dest='baseDirectory', type=str, help='Name of base directory containing timing trace files for different schedulers')
    parser.add_argument('-o', dest='outFileName', help='CSV File name to dump trace results into')
    parser.add_argument('-w', dest='workload', help='Workload type, options "LOW" or "HIGH"', required=True)
    parser.add_argument('-m', dest='min', default=0, help='Enter 1 for min, 0 for average')
    return parser


if __name__ == '__main__':
    argparser = generate_argparser()
    args = argparser.parse_args()
    BASEDIR = args.baseDirectory
    OUTFILENAME = args.outFileName
    MIN = args.min
    WORKLOAD = args.workload

    ############# Edit parameters here ####################
    CPUS=3
    FFTS=1
    MMULTS=1
    SCHEDS=["SIMPLE", "MET", "EFT", "ETF", "HEFT_RT"]
    
    if WORKLOAD == 'HIGH':
        # Use following INJ_RATES and PERIODS for High latency workload data
        INJ_RATES=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        PERIODS=[101270, 50635, 33757, 25317, 20254, 16878, 14467, 12659, 11252, 10127, 5063, 3376, 2532, 2025, 1688, 1447, 1266, 1125, 1013, 921, 844, 779, 723, 675, 633, 596, 563, 533, 506]
    elif WORKLOAD == 'LOW':
        # Use following INJ_RATES and PERIODS for Low latency workload data
        INJ_RATES=[1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950] #, 1000]
        PERIODS=[62500, 2500, 1250, 833, 625, 500, 417, 357, 313, 278, 250, 227, 208, 192, 179, 167, 156, 147, 139, 125, 114, 104, 96, 89, 83, 78, 74, 69, 66] #, 63]
    else:
        print('Wrong workload type ', WORKLOAD, ' chosen, please choose either "HIGH" or "LOW"!')
        exit()

        
    TRIALS=2 #5    # Edit here
    corelist = [' Core 1', ' Core 2', ' Core 3', ' FFT 1', ' MMULT 1']  # Edit here

    #######################################################

    with open(OUTFILENAME, 'w') as fp:
        writer = csv.writer(fp)
        header_list = ['Resource Pool', 'Scheduler', 'Injection Rate (Mbps)']
        for t in range(TRIALS):
            header_list.append('Trial ' + str(t+1))
        header_list = header_list + ['Avg. execution time / app.(ns)', 'Avg. cumulative execution time / app. (ns)', 'Avg. Scheduling overhead / app.(ns)'] + corelist
        writer.writerow(header_list)

        for m in range (MMULTS+1):  # Edit the for loop for different types of resources
            for f in range (FFTS+1):
                for c in range (CPUS):
                    for sched in SCHEDS:
                        for i in range(len(INJ_RATES)):
                            row_to_write = []
                            resPoolComb = 'C'+str(c+1)+'+F'+str(f)+'+M'+str(m)  # Edit here
                            row_to_write.append(resPoolComb)
                            row_to_write.append(sched)
                            row_to_write.append(INJ_RATES[i])
                            
                            if MIN==0:
                                avg_appexectime = 0
                                avg_cumu_appexectime = 0
                                avg_scheduletime = 0 
                            else:
                                avg_appexectime = float('inf')
                                avg_cumu_appexectime = float('inf')
                                avg_scheduletime = float('inf')

                            total_resource_util_dict = {}
                            for t in range(TRIALS):
                                file_to_parse = BASEDIR+'/'+WORKLOAD+'/trial_'+str(t+1)+'/c'+str(c+1)+'_f'+str(f)+'_m'+str(m)+'_sched-'+sched+'-p'+str(PERIODS[i])+','+str(PERIODS[i])+'/timing_trace.log' # Edit here
                                sched_file_to_parse = BASEDIR+'/'+WORKLOAD+'/trial_'+str(t+1)+'/c'+str(c+1)+'_f'+str(f)+'_m'+str(m)+'_sched-'+sched+'-p'+str(PERIODS[i])+','+str(PERIODS[i])+'/schedule_trace.log' # Edit here
                                print(file_to_parse)
                                # Call function on the file_to_parse
                                running_cumu_appexecvalue, _, app_count = cumulative_exec_only(file_to_parse)
                                running_appexecvalue, _, app_count = cumulative_exec(file_to_parse)
                                total_scheduler_overhead, _ = schedule_overhead_total(sched_file_to_parse)
                                resourceutil_dict = resource_util(file_to_parse)
                                
                                # Create a running average, or taking a minimum
                                if MIN==0:
                                    avg_appexectime += running_appexecvalue
                                    avg_cumu_appexectime += running_cumu_appexecvalue              
                                    avg_scheduletime += (total_scheduler_overhead/app_count)
                                else:
                                    avg_appexectime = min(avg_appexectime, running_appexecvalue)    
                                    avg_cumu_appexectime = min(avg_cumu_appexectime, running_cumu_appexecvalue)               
                                    avg_scheduletime = min(avg_scheduletime, (total_scheduler_overhead/app_count))  
                                
                                # Write trial result in row
                                row_to_write.append(running_cumu_appexecvalue)
                                total_resource_util_dict = add_dicts(total_resource_util_dict, resourceutil_dict)
                                #print(total_resource_util_dict)
                            
                            #Write out average in row_to_write
                            if MIN==0:
                                row_to_write.append(avg_appexectime/TRIALS)
                                row_to_write.append(avg_cumu_appexectime/TRIALS)
                                row_to_write.append(avg_scheduletime/TRIALS)
                            else:
                                row_to_write.append(avg_appexectime)  
                                row_to_write.append(avg_cumu_appexectime)
                                row_to_write.append(avg_scheduletime)
                            
                            for core in corelist:
                                row_to_write.append(total_resource_util_dict[core]/TRIALS)

                            writer.writerow(row_to_write)
