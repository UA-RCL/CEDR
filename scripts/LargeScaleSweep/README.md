# Scripts for running end-to-end CEDR large scale experiments

This folder contains scripts for running large scale CEDR experiments, and conducting analysis and plotting of the extracted experimental results.

Scripts are arranged into 3 categories:

* `1.zynq/`: This folder contains the bash scripts and files needed to run the experiments on the ZCU102 board (or the intended CEDR platform, e.g.: x86, Jetson etc.).

* `2.trace_extract/`: After the CEDR experiment runs are done, the produced log files can be parsed through this script and useful metrics can be extracted into a CSV file.

* `3.plotting/`:  This folder contains scripts to generate 2D and 3D plots from the CSV file.

Each folder contains a `README.md` files that tries to give some useful information regarding usage of the scripts, and what part of the scripts might need modification if someone wants to run and process a different set of experiments.