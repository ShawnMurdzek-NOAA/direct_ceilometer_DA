# direct_ceilometer_DA

Sandbox for developing a direct ceilometer DA algorithm using an Ensemble Square-Root Filter (EnSRF).

## Contents

- `drivers/`: Primary drivers for various cloud DA programs.
- `external/`: Submodules used by this program (e.g., [pyDA_utils](https://github.com/ShawnMurdzek-NOAA/pyDA_utils/tree/main)).
- `main/`: Main code for the direct ceilometer cloud DA.
- `misc/`: Miscellaneous files.
- `notebooks/`: Jupyter notebooks. Likely do not work anymore
- `tests/`: Simple cases for testing code changes

## Quick Start Guide

The following steps will download the program (including the required submodules), create a new Python environment (if needed), and run the test cases

1. `git clone --recurse-submodules https://github.com/ShawnMurdzek-NOAA/direct_ceilometer_DA.git`
2. Load a Python environment with the required packages. It is possible that you may already have a Python environment with all the required packages because the dependencies are somewhat common in atmospheric science (e.g., NumPy, Xarray). If not, a new environment can be created from python_environment.yml using the command `conda env create -f python_environment.yml`.
3. `cd direct_ceilometer_DA/tests`
4. `bash run_tests.sh`
