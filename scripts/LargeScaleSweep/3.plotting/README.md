# Plotting extracted CSV file from timing logs

The scripts `plt3dplot_inj.py` and `2dplot_inj.py` can be used to plot the output metrics obtained in the CSV format from the timing logs. These scripts are described below:

### `plt3dplot_inj.py`:
This script takes in the CSV file generated using `makedataframe.py` script, and plots in 3D format.

Here, (X,Y,Z) axes present (hardware configuration, injection rates, output metric), and the plot colors indicate the schedulers.


```bash
usage: plt3dplot_inj.py [-h] inputFile metricSelect

Plot a 3D plot from csv file

positional arguments:
  inputFile     Input CSV file to plot
  metricSelect  Select which metric to plot across Z-axis <CUMU, EXEC, SCHED>

optional arguments:
  -h, --help    show this help message and exit
```

The script produces output in interactive GUI window. Using this GUI window, the 3D plot can be oriented in the desired point of view and saved as a file.


### `2dplot_inj.py`:
This script uses the same CSV file as input, but produces the plot for a specific hardware configuration. 

The (X,Y) axes indicate (injection rates, output metric), and colors indicate schedulers.

```bash
usage: 2dplot_inj.py [-h] inputFile

Plot a 3D plot from csv file

positional arguments:
  inputFile   Input CSV file to plot

optional arguments:
  -h, --help  show this help message and exit
```

This script saves the 2D plots for all three types of output metrics (\<CUMU, EXEC, SCHED\>). 

The following lines of the script can be modified to select which hardware configuration results should be plotted, and if the plots should be saved or rather displayed in an interactive GUI.

```python
### Configuration specification; Edit to specify which ###
### hardware configuration metrics are needed to be plotted###
CPUS = 3
FFTS = 1
MMULTS = 1

schedlist = ['SIMPLE', 'MET', 'EFT', 'ETF', 'HEFT_RT']
schednamelist = {'SIMPLE':'RR', 'MET':'MET', 'EFT':'EFT', 'ETF':'ETF', 'HEFT_RT':'HEFT$_\mathrm{RT}$'}
schedcolordict = {'SIMPLE':'b', 'MET': 'tab:orange', 'EFT':'g', 'ETF':'r', 'HEFT_RT':'tab:purple'}
schedmarkerdict = {'SIMPLE':'o', 'MET': 'v', 'EFT':'^', 'ETF':'s', 'HEFT_RT':'d'}
##############################################################

...

# Last line of file, Comment/uncomment to either save or display plots         
plt.savefig(metric_label+'-'+csv_file_id,format='pdf', bbox_inches='tight')
plt.show()
```