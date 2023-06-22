# Script for extracting data from CEDR run logs

The script `makedataframe.py` can be used to extract useful metrics from the large number of generated log files, and dump these metrics into a CSV file for later analysis or visualization.

Usage of this file is as follows:

```bash
usage: makedataframe.py [-h] [-i BASEDIRECTORY] [-o OUTFILENAME] -w WORKLOAD
                        [-m MIN]

Python script to dump timing trace stats into a csv file!

optional arguments:
  -h, --help        show this help message and exit
  -i BASEDIRECTORY  Name of base directory containing timing trace files for
                    different schedulers
  -o OUTFILENAME    CSV File name to dump trace results into
  -w WORKLOAD       Workload type, options "LOW" or "HIGH"
  -m MIN            Enter 1 for min, 0 for average
```

Here,
* The `BASEDIRECTORY` should point to the `log_dir` folder generated through running scripts from `1.zynq/` folder, and should contains different workload (e.g. `HIGH/` or `LOW/` folders) and `trial_X` folders within the workload folders.

* The `MIN` argument's value suggests if the script should extract the minimum possible value of any output metric from all the trials, or the average over all trials. Setting it to `0` ensures average outcome.

N.B.: The extracted CSV file contains some columns with `Trial X` column header. These values correspond to `Avg. cumulative execution time / app. (ns)` values for different trials.


## Necessary edits in `makedataframe.py` for different experiments
Some of the less frequently modified input parameters are embedded into the script and might need updating if running for different set of input parameters (injection rates, number of trials, hardware configurations etc.).
These parameters are as follows:

```python
############# Edit parameters here ####################
CPUS=3
FFTS=1
MMULTS=1
SCHEDS=["SIMPLE", "MET", "EFT", "ETF", "HEFT_RT"]

if WORKLOAD == 'HIGH':
    # Use following INJ_RATES and PERIODS for High latency workload data, Update if needed
    INJ_RATES=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    PERIODS=[101270, 50635, 33757, 25317, 20254, 16878, 14467, 12659, 11252, 10127, 5063, 3376, 2532, 2025, 1688, 1447, 1266, 1125, 1013, 921, 844, 779, 723, 675, 633, 596, 563, 533, 506]
elif WORKLOAD == 'LOW':
    # Use following INJ_RATES and PERIODS for Low latency workload data, Update if needed
    INJ_RATES=[1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950] #, 1000]
    PERIODS=[62500, 2500, 1250, 833, 625, 500, 417, 357, 313, 278, 250, 227, 208, 192, 179, 167, 156, 147, 139, 125, 114, 104, 96, 89, 83, 78, 74, 69, 66] #, 63]
else:
    print('Wrong workload type ', WORKLOAD, ' chosen, please choose either "HIGH" or "LOW"!')
    exit()


TRIALS=2 #5    # Edit here
corelist = [' Core 1', ' Core 2', ' Core 3', ' FFT 1', ' MMULT 1']  # Edit here

#######################################################
```
