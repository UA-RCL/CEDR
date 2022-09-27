
"""
Python script intended to process all csv files. Hopefully all files
-p , --path_to_csv  Absolute path to the csv file with counters
"""
import numpy as np
import csv, argparse, sys

def generate_argparser():
    parser = argparse.ArgumentParser(description="Python script intended to process all csv files.")
    parser.add_argument("-p", "--path_to_csv", help="Absolute path to the csv file with counters")
    return parser

def main():
    argparser = generate_argparser()
    args = argparser.parse_args()
    csvfile = open(args.path_to_csv, 'r')
    total_papi_counter = []
    
    lines = [line for line in csvfile]
    lines = lines[1:(len(lines))]
    papi_counters = lines[0].split(";")[-1].split(",")[2:]
    if(total_papi_counter == []):
        total_papi_counter = [0]*len(papi_counters)
    lines = lines[1:(len(lines))]
    fileSum = [0]*len(papi_counters)

    for line in range(len(lines)):
      for counter in range(len(papi_counters)): 
        try:
          fileSum[counter] += int(lines[line].split(";")[-1].split(",")[2:][counter].strip(" \'\"\n"))
        except:
          print("Error on in file at line " + str(line + 3))
          print("\n")
          break
    # print(papi_counters)
    # print(fileSum)
    for counter in range(len(papi_counters)):
        total_papi_counter[counter] += fileSum[counter]
    # print(total_papi_counter)
    # print("\n")
        
    papi_counters = [counter.strip("\'\"\n ") for counter in papi_counters]
    print("These are the Papi counters:")
    print(papi_counters)
    print("Total amount for those counters:")
    print(total_papi_counter)
    csvfile.close()

    print("Placing in csv file")
    csvfile2 = open("processed.csv", 'w')
    writer = csv.writer(csvfile2)
    writer.writerow(papi_counters)
    writer.writerow(total_papi_counter)
    pass

    
if __name__ == "__main__":
    main()