# Direct Ceilometer DA Tests

## Basic Functional Test

Run `../drivers/ceilometer_obs_enkf.py` with one of the YAML files in this directory and compare the plots produced with the plots in `data/truth/`. This is also a nice sandbox for testing code changes.

| YAML file | "Truth" Plots |
|-----------|---------------|
| `S_NewEngland_2022020121_EnKF_test_input.yml` | `data/truth/` |
| `S_NewEngland_2022020121_EnKF_test_input_localization.yml` | `data/localization_truth/` |
