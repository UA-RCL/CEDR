import csv
import pandas as pd
import argparse
#import plotly.express as px
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('svg')


def generate_argparser():
    parser = argparse.ArgumentParser(description="Plot a 3D plot from csv file")
    parser.add_argument("inputFile", help="Input CSV file to plot")
    parser.add_argument("metricSelect", help="Select which metric to plot across Z-axis <CUMU, EXEC, SCHED>")
    return parser


if __name__ == "__main__":
    argparser = generate_argparser()
    args = argparser.parse_args()

    ### Configuration specification ###
    CPUS = 3
    FFTS = 0
    MMULTS = 0
    ZIPS = 0
    GPUS = 0
    WORKLOAD = 'High'
    TRIALS = 2
    schedlist = {'SIMPLE':1, 'MET':2, 'ETF':3}
    schedmarkerlist = {'SIMPLE':'o', 'MET':'o', 'ETF':'o'}
    schednamelist = ['RR', 'MET', 'ETF']
    #colorlist = {' Core 1':'green', ' Core 2': 'blue', ' Core 3': 'red', ' FFT 1':'black', ' MMULT 1': 'orange','ZIP 1: 'purple'}
    viridis = cm.get_cmap('tab10',10)
    #print(viridis(1))
    #exit(0)
    configlist = []
    configlist_tick = []
    for c in range (CPUS):
        for f in range (FFTS+1):
            for m in range(MMULTS+1):
              for z in range (ZIPS+1):
                for g in range (GPUS+1):
                  if GPUS==0:
                    configlist.append('C'+str(c+1)+'+F'+str(f)+'+M'+str(m)+'+Z'+str(z))
                    configlist_tick.append('C'+str(c+1)+'\nF'+str(f)+'\nM'+str(m)+'\nZ'+str(z))
                  else:
                    configlist.append('C'+str(c+1)+'+F'+str(f)+'+M'+str(m)+'+Z'+str(z)+'+G'+str(g))
                    configlist_tick.append('C'+str(c+1)+'\nG'+str(g))
    print(configlist)
    metrics = ['Avg. cumulative execution time / app. (ns)', 'Avg. execution time / app.(ns)', 'Avg. Scheduling overhead / app.(ns)']
    metricnames = ['Avg. Cumulative Execution\nTime / App.(ms)', 'Avg. Execution Time / App.(ms)', 'Avg. Scheduling Overhead / App.(ms)']

    metricSelect = args.metricSelect

    if metricSelect == 'CUMU':
        metricselect = 0
    elif metricSelect == 'EXEC':
        metricselect = 1
    elif metricSelect == 'SCHED':
        metricselect = 2
    else:
        print("Invalid metric '", metricSelect , "' selected, please select from [CUMU, EXEC, SCHED]")
        exit()
    #metric = 'Avg. execution time / app.(ns)'
    #metric = 'Avg. Scheduling overhead / app.(ns)'
    ###################################


    df = pd.read_csv(args.inputFile, sep=',')
    print(df)

    fig = plt.figure() #figsize=(10,6))
    ax = plt.axes(projection='3d')
    configcount = 1
    i = 0
    for config in configlist:
        ###### Filter by resource pool configuration
        df_rp = df.loc[df['Resource Pool'] == config]
        print(df_rp)

        ###### Keep only relevant information
        df_rp_sc_metric = df_rp.filter(items=['Scheduler', 'Injection Rate (Mbps)', metrics[metricselect]])
        print(df_rp_sc_metric)


        # Find list of available injection rates
        inj_rates = list(df_rp_sc_metric['Injection Rate (Mbps)'])
        inj_rates = sorted(list(set(inj_rates)))
        print(inj_rates)

        for sched in schedlist.keys():
            ###### Extract scheduler values as X axis
            """ Right now defined on top """
            x = inj_rates
            print (x)

            ###### Fix resource pool address array
            """ Right now defined on top """
            y = list(np.ones(len(inj_rates))*configcount)
            print(y)
            ###### Extract corresponding Z-axis metrics values
            z = []
            for inj in x:
                z.append(df_rp_sc_metric.loc[(df_rp_sc_metric['Scheduler'] == sched) & (df_rp_sc_metric['Injection Rate (Mbps)'] == inj)] [metrics[metricselect]].values[0])
            print(z)

            z = [dz/1000000 for dz in z]
            print(z)

            # 3d plot
            #fig = plt.figure()
            #ax = plt.axes(projection='3d')
            #ax.plot3D(x,y,z,'gray')
            if i == 0:
                p = ax.scatter3D(x,y,z, alpha=1, c=viridis(schedlist[sched]-1), s=40, marker=schedmarkerlist[sched], label=schednamelist[schedlist[sched]-1])
            else:
                p = ax.scatter3D(x,y,z, alpha=1, c=viridis(schedlist[sched]-1), s=40, marker=schedmarkerlist[sched])
        configcount += 1
        i+=1

    # Custom plotting to identify outlier
    # High
    """
    x_value = 40 # Injection rate
    y_value = 11 # hw configuration c1_f1_m1
    config = 'C3+F1+M0'
    z_value = (df.loc[(df['Resource Pool'] == config) & (df['Scheduler']=='EFT') & (df['Injection Rate (Mbps)']==x_value)] [metrics[metricselect]].values[0]) / 1000000
    print(z_value)
    ax.scatter3D(x_value, y_value, z_value, alpha=0.6, s=500, marker='s', facecolors=[1.0, 1.0, 1.0, 0.5], edgecolor='red', linewidth=3)
    
    
    #Low
    x_value = 325 # Injection rate
    y_value = 4 # hw configuration c1_f1_m1
    config = 'C1+F1+M1'
    z_value = (df.loc[(df['Resource Pool'] == config) & (df['Scheduler']=='HEFT_RT') & (df['Injection Rate (Mbps)']==x_value)] [metrics[metricselect]].values[0]) / 1000000
    print(z_value)
    ax.scatter3D(x_value, y_value, z_value, alpha=0.6, s=500, marker='s', facecolors=[1.0, 1.0, 1.0, 0.5], edgecolor='red', linewidth=3)
    """

    # Set axis tick names
    #plt.xticks(x, list(schedlist.keys()))
    #ax.set_xticks(x)
    #ax.set_xticklabels(x, fontsize= 18)
    #ax.set_xticklabels(fontsize=18)
    # ax.set_xticklabels(list(schedlist.keys()), {'fontsize':16})
    #plt.yticks(list(np.arange(1,configcount)),configlist_tick )
    ax.set_xticks(np.arange(0,max(x)+1, max(x)/4))
    ax.set_yticks(np.arange(1,configcount))
    ax.set_yticklabels(configlist_tick, fontsize=15)
    ax.tick_params(axis='z', which='major', labelsize=20)
    ax.tick_params(axis='x', which='major', labelsize=20)

    ax.set_xlabel("\n\n\nInjection Rate\n(Mbps)", fontsize=24, fontweight='bold')
    ax.set_ylabel("\n\n\n\n\n\nHardware Configurations", fontsize=24, fontweight='bold')
    ax.set_zlabel("\n\n"+metricnames[metricselect], fontsize=22, fontweight='bold')
    plt.rc('axes', labelsize=50)
    #plt.legend(fontsize=20, ncol=5, loc='upper center')
    ax.legend(markerscale=3, fontsize=20, ncol=5, loc='upper center')
    plt.show()
    plt.savefig("dse_sched.png",dpi=1200)
