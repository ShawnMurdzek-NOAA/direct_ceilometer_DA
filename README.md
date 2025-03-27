# direct_ceilometer_DA

Sandbox for developing a direct ceilometer DA algorithm using an Ensemble Square-Root Filter (EnSRF).

## Contents

- `external/`: Submodules used by this program (e.g., [pyDA_utils](https://github.com/ShawnMurdzek-NOAA/pyDA_utils/tree/main)).
- `main/`: Main code for the direct ceilometer cloud DA.
- `misc/`: Miscellaneous files.
- `notebooks/`: Jupyter notebooks. Likely do not work anymore
- `tests/`: Simple cases for testing code changes

## Quick Start Guide

Start by downloading the code from GitHub, including the required submodules:

`git clone --recurse-submodules https://github.com/ShawnMurdzek-NOAA/direct_ceilometer_DA.git`

Next, configure the required Python environment. If conda is enabled, a new environment can be created by running the following, with `{ENV_PREFIX}` replaced with the desired install location for the new environment:

```
cd direct_ceilometer_DA
conda env create -f python_environment.yml --prefix {ENV_PREFIX}
conda activate {ENV_PREFIX}
```

The program requires a single YAML input file. Examples can be found in the `tests/` directory and more details can be found in `README_inputs.md`. The program can be run using the following command, with `{YAML}` replaced with the input YAML file name:

`python ceilometer_obs_enkf.py {YAML}`

### Running the test cases

Test cases, along with sample output, are included in the `tests/` directory. To run these tests, activate the required Python environment (see above), then run the following:

```
cd tests
bash run_tests.sh
```

The top portion of `run_tests.sh` can be edited to control which test cases are run.
