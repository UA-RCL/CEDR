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
            resource_name = row[4].split(':')[1]
            start_time = row[5].split(':')[1]
            stop_time = row[6].split(':')[1]
            execution_time = row[7].split(':')[1]

            schedule_event = ScheduleEvent(resource_name, (int(start_time)), (int(stop_time)), (int(execution_time)))

            if resource_name in proc_schedules:
                proc_schedules[resource_name].append(schedule_event)
            else:
                proc_schedules[resource_name] = [schedule_event]

        start_dict = {}
        stop_dict = {}
        exectime_dict = {}
        corelist = [' cpu1', ' cpu2', ' cpu3', ' fft1',' fft2',' fft3',' fft4',' fft5',' fft6', ' fft7', ' fft8', ' mmult1', ' mmult2', ' zip1', ' zip2']  # Edit here
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

def apprun_time(FILENAME):
    num_apps = 0
    app_runtime = 0
    with open(FILENAME, newline='') as f:
        reader = csv.reader(f)
        for row in reader: 
            app_id= int(row[0].split(':')[1])
            app_name= (row[1].split(':')[1].strip())
            app_runtime=app_runtime + int (row[5].split(':')[1].strip())    
            num_apps = num_apps + 1
               
    return app_runtime/num_apps, 0, num_apps             
      
  #The runtime of the applications{app_runtime) over the number of applications{num_apps} gives the average application runtime/num_apps   

def cumulative_exec(FILENAME):

    app_starts = {}
    app_ends = {}

    # Determine the start and end time of each application
    with open(FILENAME, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            app_id = int(row[0].split(':')[1].strip())
            start_time = int(row[5].split(':')[1].strip())
            end_time = int(row[6].split(':')[1].strip())

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
            start_time = int(row[5].split(':')[1].strip())
            end_time = int(row[6].split(':')[1].strip())

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
    parser.add_argument('-t', dest='trial', default=3, help='Number of trials to use')
    parser.add_argument('-r', dest='injectionRateCount', default=13, help='Number of injection rates used for the tests')
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
    FFTS=8
    MMULTS=0
    ZIPS=0
    
    SCHEDS=["SIMPLE", "MET", "EFT", "ETF"]
    
    if WORKLOAD == 'HIGH':
        # Use following INJ_RATES and PERIODS for High latency workload data
        #INJ_RATES=[10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000]    
        #WF_PERIODS=[346875,173438,115625,86719,69375,57813,49554,43359,38542,34688,17344,11563,8672,6938,5781,4955,4336,3854,3469,3153,2891,2668,2478,2313,2168,2040,1927,1826,1734]
        #PD_PERIODS=[433594,216797,144531,108398,86719,72266,61942,54199,48177,43359,21680,14453,10840,8672,7227,6194,5420,4818,4336,3942,3613,3335,3097,2891,2710,2551,2409,2282,2168]
        
        INJ_RATES=[2000,1500,1000,500,100,10,1800,1300,800,300,50,1700,1200,700,200,70,30,80,20,1900,1600,1100,600,90,40,1400,900,400,60]    
        WF_PERIODS=[1734,2313,3469,6938,34688,346875,1927,2668,4336,11563,69375,2040,2891,4955,17344,49554,115625,43359,173438,1826,2168,3153,5781,38542,86719,2478,3854,8672,57813]
        PD_PERIODS=[2168,2891,4336,8672,43359,433594,2409,3335,5420,14453,86719,2551,3613,6194,21680,61942,144531,54199,216797,2282,2710,3942,7227,48177,108398,3097,4818,10840,72266]
        
    elif WORKLOAD == 'LOW':
        print('Low workload is not specified for this setup')
    else:
        print('Wrong workload type ', WORKLOAD, ' chosen, please choose either "HIGH" or "LOW"!')
        exit()

    INJ_COUNT=int(args.injectionRateCount) #13
    TRIALS=int(args.trial)
    corelist = [' cpu1', ' cpu2', ' cpu3', ' fft1',' fft2',' fft3',' fft4',' fft5',' fft6', ' fft7', ' fft8', ' mmult1', ' mmult2', ' zip1', ' zip2']  # Edit here

    #######################################################

    with open(OUTFILENAME, 'w') as fp:
        writer = csv.writer(fp)
        header_list = ['Resource Pool', 'Scheduler', 'Injection Rate (Mbps)']
        for t in range(TRIALS):
            header_list.append('Trial ' + str(t+1))
        header_list = header_list + ['Avg. execution time / app.(ns)', 'Avg. cumulative execution time / app. (ns)', 'Avg. Scheduling overhead / app.(ns)'] + corelist
        writer.writerow(header_list)
        
        col1=[]
        col2=[]
        col3=[]
        col4=[]
        col5=[]
        col6=[]
        col7=[]
        foundCount=[]
        for z in range (ZIPS+1): 
            for m in range (MMULTS+1):  # Edit the for loop for different types of resources
                for f in range (FFTS+1):
                    for c in range (CPUS):
                        for sched in SCHEDS:
                            for i in range(INJ_COUNT):#range(len(INJ_RATES)):
                                rFound=False
                                resPoolComb = (c+1)+f+m+z # Edit here
                                
                                if len(col1) == 0:
                                    col1.append(resPoolComb)
                                    col2.append(sched)
                                    col3.append(INJ_RATES[i])
                                    foundCount.append(1)
                                else:
                                    for t in range(len(col1)):
                                        if (resPoolComb == col1[t]) and (sched == col2[t]) and (INJ_RATES[i] == col3[t]):
                                            rFound=True
                                            foundWhere=t
                                            foundCount[foundWhere]+=1
                                            break
                                    if rFound == False:
                                        col1.append(resPoolComb)
                                        col2.append(sched)
                                        col3.append(INJ_RATES[i])
                                        foundCount.append(1)
                                
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
                                    file_to_parse = BASEDIR+'/'+WORKLOAD+'/trial_'+str(t+1)+'/c'+str(c+1)+'_f'+str(f)+'_m'+str(m)+'_z'+str(z)+'_sched-'+sched+'-p0,'+str(WF_PERIODS[i])+','+str(WF_PERIODS[i])+','+str(PD_PERIODS[i])+'/timing_trace.log' # Edit here
                                    sched_file_to_parse = BASEDIR+'/'+WORKLOAD+'/trial_'+str(t+1)+'/c'+str(c+1)+'_f'+str(f)+'_m'+str(m)+'_z'+str(z)+'_sched-'+sched+'-p0,'+str(WF_PERIODS[i])+','+str(WF_PERIODS[i])+','+str(PD_PERIODS[i])+'/schedule_trace.log' # Edit here
                                    apprun_file_to_parse= BASEDIR+'/'+WORKLOAD+'/trial_'+str(t+1)+'/c'+str(c+1)+'_f'+str(f)+'_m'+str(m)+'_z'+str(z)+'_sched-'+sched+'-p0,'+str(WF_PERIODS[i])+','+str(WF_PERIODS[i])+','+str(PD_PERIODS[i])+'/appruntime_trace.log'
                                    print(file_to_parse)
                                    # Call function on the file_to_parse
                                    running_cumu_appexecvalue, _, app_count = cumulative_exec_only(file_to_parse)
                                    running_appexecvalue, _, app_count = apprun_time(apprun_file_to_parse)
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
                                    if rFound:
                                       col4[foundWhere] = ( ((foundCount[foundWhere]-1)*col4[foundWhere]) + running_cumu_appexecvalue ) / foundCount[foundWhere]
                                    else:
                                        col4.append(running_cumu_appexecvalue)
                                    total_resource_util_dict = add_dicts(total_resource_util_dict, resourceutil_dict)
                                    #print(total_resource_util_dict)
                                
                                #Write out average in row_to_write
                                if MIN==0:
                                    if rFound:
                                        col5[foundWhere] = ( ((foundCount[foundWhere]-1)*col5[foundWhere]) + (avg_appexectime/TRIALS) ) / foundCount[foundWhere]
                                        col6[foundWhere] = ( ((foundCount[foundWhere]-1)*col6[foundWhere]) + (avg_cumu_appexectime/TRIALS) ) / foundCount[foundWhere]
                                        col7[foundWhere] = ( ((foundCount[foundWhere]-1)*col7[foundWhere]) + (avg_scheduletime/TRIALS) ) / foundCount[foundWhere]
                                    else:
                                        col5.append(avg_appexectime/TRIALS)
                                        col6.append(avg_cumu_appexectime/TRIALS)
                                        col7.append(avg_scheduletime/TRIALS)
                                else:
                                    if rFound:
                                        col5[foundWhere] = ( ((foundCount[foundWhere]-1)*col5[foundWhere]) + (avg_appexectime) ) / foundCount[foundWhere]
                                        col6[foundWhere] = ( ((foundCount[foundWhere]-1)*col6[foundWhere]) + (avg_cumu_appexectime) ) / foundCount[foundWhere]
                                        col7[foundWhere] = ( ((foundCount[foundWhere]-1)*col7[foundWhere]) + (avg_scheduletime) ) / foundCount[foundWhere]
                                    else:
                                        col5.append(avg_appexectime)
                                        col6.append(avg_cumu_appexectime)
                                        col7.append(avg_scheduletime)
        #print(col1)
        for i in range(len(col1)):
            row_to_write = []
            row_to_write.append(col1[i])
            row_to_write.append(col2[i])
            row_to_write.append(col3[i])
            row_to_write.append(col4[i])
            row_to_write.append(col5[i])
            row_to_write.append(col6[i])
            row_to_write.append(col7[i])
            for core in corelist:
                row_to_write.append(0)
            writer.writerow(row_to_write)
