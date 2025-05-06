# direct_ceilometer_DA

Program for directly assimilating ceilometer cloud cover observations using a data assimilation (DA) technique known as the Ensemble Square-Root Filter (EnSRF).

## Contents

- `main/`: Main code for the direct ceilometer cloud DA program.
- `misc/`: Miscellaneous scripts. Most relate to plotting.
- `pyDA_utils/`: Submodule. Source code can be found here at [pyDA_utils](https://github.com/ShawnMurdzek-NOAA/pyDA_utils/tree/main)
- `tests/`: Automated unit tests as well as a simple test case.

## Quick Start Guide

Start by downloading the code from GitHub, including the required submodules:

`git clone --recurse-submodules https://github.com/ShawnMurdzek-NOAA/direct_ceilometer_DA.git`

Next, configure the required Python environment. If conda is enabled, a new environment can be created by running the following, with `{ENV_PREFIX}` replaced with the desired install location for the new environment:

```
cd direct_ceilometer_DA
conda env create -f python_environment.yml --prefix {ENV_PREFIX}
conda activate {ENV_PREFIX}
```

The program requires a single YAML input file. An example is provided here: `tests/sample.yml`, and more details regarding YAML file options can be found in `README_inputs.md`. Assuming that the Python environment is configured correctly (see above), the test case can be run using the following command:

`python ceilometer_obs_enkf.py ./tests/sample.yml`

This test case uses the following inputs:
- `tests/sample_data/mpas/memXXX/mpasout.2024-05-27_04:00:00.TEST.nc`: MPAS ensemble background files. Files come from MPAS run in a limited-area configuration centered on CONUS with 12-km mesh spacing.
- `tests/sample_data/bufr/2024052704.rap.t04z.prepbufr.csv`:  BUFR ceilometer observations. 

If the program runs successfully, the following files will be created:
- `tests/sample_data/mpas/memXXX/mpasout.DA.2024-05-27_04:00:00.TEST.nc`: MPAS ensemble analysis files
- `EnKF_diag.csv`: DA diagnostics.

## Supported Input File Types

### Ensemble Background

Ensemble I/O is handled using `main/ens_io.py`. The only ensemble file format the is fully supported is MPAS netCDF output. A function exists in `main/ens_io.py` for reading UPP GRIB2 output, but there is currently no function for writing analysis files in UPP GRIB2 format.

### Observations

Observation I/O is handled using `main/obs_io.py`. Observations must be in the CSV format produced by [prepbufr_decoder](https://github.com/ShawnMurdzek-NOAA/prepbufr_decoder), which converts prepBUFR files to CSVs.

## DA Diagnostics

Basic DA diagnostics are output from the program in a CSV file. The diagnostic file contains a separate line for each observation that was assimilated, with the following fields:

- Observation height (m AGL)
- Observation longitude (deg E)
- Observation latitude (deg N)
- Observed cloud fraction (%)
- O-B values for each ensemble member (labeled ombN, where N is the ensemble member). These values are in %.
- O-A values for each ensemble member (labeled omaN, where N is the ensemble member). These values are in %.

## Visualization

Current visualization capabilities are rather limited. The following scripts may be helpful for various types of simple visualization.

### MPAS NetCDF Output

There are no scripts included in this repository for visualizing raw MPAS netCDF output. One option for visualizing raw MPAS netCDF output is the [plot_mpas_hcrsxn.py](https://github.com/ShawnMurdzek-NOAA/py_scripts/blob/main/mpas/plot_mpas_hcrsxn.py) program in py_scripts. This script can plot both horizontal cross sections of fields from a single MPAS output file as well as differences between two different MPAS output files.

### Observations

The `misc/plot_bufr_cloud_obs` program can be used to plot ceilometer observations in the CSV format mentioned above. This program can be run by executing the following command, with `{CSV FILE NAME}` replaced with the CSV file containing the ceilometer observations:

`python misc/plot_bufr_cloud_obs/plot_bufr_cloud_obs.py {CSV FILE NAME} misc/plot_bufr_cloud_obs/param_bufr_cloud_obs.yml`
