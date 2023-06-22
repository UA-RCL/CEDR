import sys

"""
TODO for next revision: 
1. Add input arguments for list of worker thread types and counts, as well as schedulers.
This would minimize the need to edit script for generating different CEDR configurations.
2. Instead of using stdout to put a string onto a file, construct a dictionary based on
input arguments and dump into file using json.dumps(). More efficient and less script editing
needed for different CEDR configurations.
"""

if __name__=="__main__":
    print("Starting creating daemon_config_files")

    SCHEDS = ["SIMPLE", "EFT", "ETF", "MET"] #, "HEFT_RT"]
    CPUS = 3
    FFTS = 8
    MMULTS = 2
    ZIPS = 2

    original_stdout = sys.stdout

    for sched in SCHEDS:
        for zipp in range (ZIPS+1):
            for mmult in range (MMULTS+1):
                for fft in range (FFTS+1):
                    for cpu in range(CPUS):
                        filename = "daemon_config-"+"c"+str(cpu+1)+"_f"+str(fft)+"_m"+str(mmult)+"_z"+str(zipp)+"-"+str(sched)+".json"
                        with open(filename, 'w') as f:
                            sys.stdout = f

                            print('{\n'
                                    '\t"Worker Threads": {\n'
                                        '\t\t"cpu": '+str(cpu+1)+',\n'
                                        '\t\t"fft": '+str(fft)+',\n'
                                        '\t\t"mmult": '+str(mmult)+',\n'
                                        '\t\t"zip": '+str(zipp)+'\n'
                                     '\t},\n'

                                    '\t"Features": {\n'
                                        '\t\t"Cache Schedules": false,\n'
                                        '\t\t"Enable Queueing": true,\n'
                                        '\t\t"Use PAPI": true,\n'
                                        '\t\t"Loosen Thread Permissions": true,\n'
                                        '\t\t"Fixed Periodic Injection": true,\n'
                                        '\t\t"Exit When Idle": true\n'
                                    '\t},\n'

                                    '\t"PAPI Counters": [\n'
                                        '\t\t"perf::INSTRUCTIONS",\n'
                                        '\t\t"perf::CYCLES",\n'
                                        '\t\t"perf::BRANCHES",\n'
                                        '\t\t"perf::BRANCH-MISSES",\n'
                                        '\t\t"perf::L1-DCACHE-LOADS",\n'
                                        '\t\t"perf::L1-DCACHE-LOAD-MISSES"\n'
                                    '\t],\n'

                                    '\t"Scheduler": "'+str(sched)+'",\n'
                                    '\t"Random Seed": 0\n'
                                '}\n')



