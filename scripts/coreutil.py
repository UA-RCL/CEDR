import csv
import argparse
from collections import namedtuple


ScheduleEvent = namedtuple("ScheduleEvent", 'resource start stop exectime')


def generate_argparser():
    parser = argparse.ArgumentParser(description="A tool for extracting core utilization from trace files")
    parser.add_argument("inputFile", help="Input trace file for the core utilization to be extracted from")
    return parser


if __name__ == '__main__':
    argparser = generate_argparser()
    args = argparser.parse_args()

    proc_schedules = {}
    cores_seen = {}
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            job_id = row[0].split(':')[1].strip()
            frame_id = row[1].split(':')[1]
            app_name = row[2].split(':')[1]
            task_id = row[3].split(':')[1]
            task_name = row[4].split(':')[1]
            resource_name = row[5].split(':')[1]
            start_time = row[6].split(':')[1]
            stop_time = row[7].split(':')[1]
            execution_time = row[8].split(':')[1]

            #print(job_id, frame_id, app_name, resource_name, "\n")
            schedule_event = ScheduleEvent(resource_name, (int(start_time)), (int(stop_time)), int(execution_time))
            if resource_name in proc_schedules:
                proc_schedules[resource_name].append(schedule_event)
            else:
                proc_schedules[resource_name] = [schedule_event]
        #print(list(proc_schedules.keys()))
        start_dict = {}
        stop_dict = {}
        exectime_dict = {}
        corelist = list(proc_schedules.keys())
        # Adding all time points in three dictionaries
        for core in corelist:
            for sched in proc_schedules[core]:
                if core in start_dict.keys():
                    start_dict[str(core)].append(sched[1])
                    stop_dict[str(core)].append(sched[2])
                    exectime_dict[str(core)].append(sched[3])
                else:
                    start_dict[str(core)] = [sched[1]]
                    stop_dict[str(core)] = [sched[2]]
                    exectime_dict[str(core)] = [sched[3]]

        # Finding minimum in start_dict, maximum in stop_dict, and addition of list in exectime_dict
        core_util_dict = {}
        print("The utilization percentages of ", len(corelist), " threads are\n")
        for core in corelist:
            start_dict[str(core)] = min(start_dict[str(core)])
            stop_dict[str(core)] = max(stop_dict[str(core)])
            exectime_dict[str(core)] = sum(exectime_dict[str(core)])
        start_minimum = min(start_dict.values())
        stop_maximum = max(stop_dict.values())

        for core in corelist:
            core_util_dict[str(core)] = exectime_dict[str(core)] / (stop_maximum - start_minimum)
        #print(start_dict, stop_dict, start_minimum, stop_maximum)
            #core_util_dict[str(core)] = exectime_dict[str(core)] / (stop_dict[str(core)] - start_dict[str(core)])
            print(core, " => ", core_util_dict[str(core)]*100, "%\n")

        print("Overall resource utilization percentage, ", sum(core_util_dict.values())*100 / len(corelist), "%")

        #print("\n ", min(exectime_dict.values()))

        #print(test_dict,"\n")


