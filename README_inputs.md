# Input YAML File Description

The inputs for `ceilometer_obs_enkf.py` come from a single YAML file. Examples of these YAML files can be found in the `tests/` directory. This README includes descriptions of the various fields in these input YAML files.

## Input data sources

- **str_format**: Ensemble member output files. Must include a {num} placeholder for the ensemble member number and a {lev} placeholder for the level type (prslev or natlev).
- **prslev_vars**: Pressure-level variables to include. Naming convention follows that used in Xarray when using pyNIO as the engine.
- **nmem**: Number of ensemble members
- **save_to_nc**: Option to save ensemble member output to a netCDF file for easier and quicker I/O in the future. If set to True, the program will check to see if the netCDF file specified in **save_ens_nc** exists. If it does exist, ensemble member output is read in from that netCDF file rather than the files specified using **str_format**. If the netCDF file does not exist, then **str_format** is used and ensemble member output is saved to **subset_ens_nc**.
- **subset_ens_nc**: NetCDF file where ensemble member output is saved to.
- **bufr_fname**: BUFR CSV file containing the observations used for DA.

## Subset domain

Rather than running the DA algorithm over the entire domain, the program instead subsets the data to a subdomain first to reduce computational cost. This domain is configured using the following:

- **min_lon**: Minimum longitude (deg E)
- **max_lon**: Maximum longitude (deg E)
- **min_lat**: Minimum latitude (deg N)
- **max_lat**: Maximum latitude (deg N)
- **z_ind**: Model native vertical levels to include
