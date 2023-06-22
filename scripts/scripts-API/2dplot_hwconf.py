import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import argparse

def generate_argparser():
    parser = argparse.ArgumentParser(description="Plot a 3D plot from csv file")
    parser.add_argument("inputFile", help="Input CSV file to plot")
    return parser

if __name__ == '__main__':
    argparser = generate_argparser()
    args = argparser.parse_args()

    ### Configuration specification; Edit to specify which ###
    ### hardware configuration metrics are needed to be plotted###
    CPUS = 3
    FFTS = 8
    MMULTS = 0

    schedlist = ['SIMPLE', 'MET', 'EFT', 'ETF'] #, 'HEFT_RT']
    schednamelist = {'SIMPLE':'RR', 'MET':'MET', 'EFT':'EFT', 'ETF':'ETF', 'HEFT_RT':'HEFT$_\mathrm{RT}$'}
    schedcolordict = {'SIMPLE':'b', 'MET': 'tab:orange', 'EFT':'g', 'ETF':'r', 'HEFT_RT':'tab:purple'}
    schedmarkerdict = {'SIMPLE':'o', 'MET': 'v', 'EFT':'^', 'ETF':'s', 'HEFT_RT':'d'}
    ##############################################################
    hwconfig = CPUS+FFTS#'C'+str(CPUS)+'+F'+str(FFTS)+'+M'+str(MMULTS)  

    df = pd.read_csv(args.inputFile, sep=',')
    print(df)
#    df = df.loc[df['Resource Pool'] == hwconfig]
    df = df.loc[df['Injection Rate (Mbps)'] == 1000]

    #metrics = ['Avg. cumulative execution time / app. (ns)', 'Avg. execution time / app.(ns)', 'Avg. Scheduling overhead / app.(ns)']
    metrics = ['Avg. cumulative execution time / app. (ns)', 'Avg. execution time / app.(ns)', 'Avg. Scheduling overhead / app.(ns)']
    metric_names = ['Avg. cumulative\nexecution time / app. (ms)', 'Avg. execution time / app.(ms)', 'Avg. Scheduling overhead / app.(ms)']
    metric_labels = ['cumu_exec', 'exec', 'sched_overhead']

    for metric, metric_name, metric_label in zip(metrics, metric_names, metric_labels):
        plt.figure(figsize=(20,10))
        for sched in schedlist:
            fdf = df.loc[df['Scheduler'] == sched]
            fdf = fdf.filter(items=['Resource Pool', 'Scheduler', 'Injection Rate (Mbps)', metric])
            print(fdf)

            # Find the injection rates from CSV file
            inj_rates = list(fdf['Resource Pool'])
            print(inj_rates)

            metric_vals = []
            for inj_rate in inj_rates:
                #metric_vals.append(fdf.loc[fdf['Injection Rate (Mbps)'] == inj_rate][metric].values[0])
                metric_vals.append(fdf.loc[fdf['Resource Pool'] == inj_rate][metric].values[0])

            metric_vals = [x/1000000 for x in metric_vals]  # Converting into ms
            print('For scheduler ', sched, ', values are')
            print(metric_vals)

            plt.plot(inj_rates, metric_vals, c=schedcolordict[sched])
            plt.scatter(inj_rates, metric_vals, c=schedcolordict[sched], s=100, marker=schedmarkerdict[sched], label=schednamelist[sched])
        plt.xlabel('Resource Pool', fontsize=28, fontweight='bold')
        plt.ylabel(metric_name, fontsize=28, fontweight='bold')
        plt.xticks(fontsize=24)
        #plt.xticks(hw_config_ticks, configlist_tick, fontsize=24)
        #plt.xticks(inj_rates)
        plt.yticks(fontsize=24)
        plt.grid()
        plt.legend(fontsize=24)

        #workload_id = args.inputFile.replace('.csv','').replace('largesweep_results_','').replace('/','_')
        #plt.savefig('hwsweep_'+metric_label+'-'+workload_id+'.pdf',format='pdf', bbox_inches='tight')
        csv_file_id = args.inputFile.replace('.csv','.pdf').split('/')[-1]
        # Last line of file, Comment/uncomment to either save or display plots         
        plt.savefig(metric_label+'-'+csv_file_id,format='pdf', bbox_inches='tight')
        #plt.show()
