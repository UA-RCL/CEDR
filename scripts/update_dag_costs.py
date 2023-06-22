import argparse
import csv
import json
import os

def generate_argparser():
    parser = argparse.ArgumentParser(description="JSON profiling modifier - modifies a given application's JSON DAG based on the profiling results from a CEDR timing log")
    
    parser.add_argument("-j", "--json", help="The JSON DAG to be modified")
    parser.add_argument("-p", "--profiling-results", help="The timing log from CEDR that gives the profiling results for each node")

    return parser

if __name__ == "__main__":
    parser = generate_argparser()
    args = parser.parse_args()

    with open(args.json) as app_json:
        app = json.load(app_json)

    with open(args.profiling_results) as app_logs:
        reader = csv.reader(app_logs)
        for row in reader:
            app_name = row[2].split(':')[1].strip()
            task_name = row[4].split(':')[1].strip()
            resource_name = row[5].split(':')[1].strip()
            task_exec = row[8].split(':')[1].strip()

            # The actual name of the task in the DAG does not have the app name prepended
            print(app_name)
            print(task_name)

            task_name = task_name.replace(app_name + "_", "")

            print(task_name)

            if "Core" in resource_name:
                resource_type = "cpu"
            elif "FFT" in resource_name:
                resource_type = "fft"
            elif "MMULT" in resource_name:
                resource_type = "mmult"
            else:
                print(f"I do not know which resource type is associated with \"{resource_name}\"")

            for platform in app["DAG"][task_name]["platforms"]:
                if platform["name"] == resource_type:
                    platform["nodecost"] = int(task_exec)
    
    # https://stackoverflow.com/a/3548689
    with open(f"{os.path.splitext(args.json)[0]}-modified.json", "w") as out_file:
        json.dump(app, out_file, indent=4)
