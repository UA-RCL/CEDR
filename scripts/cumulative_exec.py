import numpy as np
import csv
import argparse
import sys

def generate_argparser():
    parser = argparse.ArgumentParser(description="Calculate a cumulative execution time value similar to DS3 from a CEDR trace")
    parser.add_argument("inputFile", help="Input trace file to be analyzed")
    return parser

if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    app_starts = {}
    app_ends = {}

    # Determine the start and end time of each application
    with open(args.inputFile, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            app_id = row[0].split(':')[1].strip()
            start_time = row[6].split(':')[1].strip()
            end_time = row[7].split(':')[1].strip()

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

    print(f"The cumulative execution time was {cumulative_exec} over {num_jobs} jobs, giving an average execution of {cumulative_exec / num_jobs}")

