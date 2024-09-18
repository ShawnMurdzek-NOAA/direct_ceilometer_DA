#!/bin/sh

#SBATCH -A nrtrr
#SBATCH -t 08:00:00
#SBATCH --nodes=1 --ntasks=1
#SBATCH --mem=10GB

# Note: Sometimes running this script returns the following error: 
#
# _tkinter.TclError: couldn't connect to display "localhost:11.0"
#
# I'm not entirely sure why this happens sometimes. It doesn't seem to pertain to a certain partition
# (I've had the code run successfully and return an error on sjet in separate instances). Submitting
# during a Jet session using port forwarding (e.g., with "jet_tunnel") with sjet appears to work.

date
. ~/.bashrc
my_py

python single_ceilometer_ob_enkf.py /lfs5/BMC/wrfruc/murdzek/src/py_scripts/cloud_da/cases/S_NewEngland_2022020121_single_ob_test/default_test/S_NewEngland_2022020121_single_ob_DA_input.yml

report-mem
date
