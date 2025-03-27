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

- **min_lon**: Minimum longitude (deg E).
- **max_lon**: Maximum longitude (deg E).
- **min_lat**: Minimum latitude (deg N).
- **max_lat**: Maximum latitude (deg N).
- **z_ind**: Model native vertical levels to include.

## DA options

- **state_vars**: Variable to include in the state vector. It is assumed that these variables are 3D. Naming convention follows that used in Xarray when using pyNIO as the engine.
- **perform_da**: Option to perform DA. Setting to False only plots the background fields. If only plotting the background fields, it is also helpful to set **plot_stat_config/plot_bgd_once** to True.
- **do_bec**: Option to compute the full background error covariance matrix (deprecated, do not use).
- **ob_sel**: Observations used for DA. Includes multiple levels:
  - Experiment name. To use all observations in **bufr_fname**, this level should be "<experiment name>: entire file", with <experiment name> replaced with the appropriate experiment name
    - Station IDs (e.g., KHFD)
      - Observation indices to use (starting from 0).
- **ob_var**: Observation error variance for the cloud cover observations (units: %^2).
- **redo_hofx**: Option to recompute the forward operator after each observation is assimilated.
- **hofx_kw**: Keyword arguments passed to `main.cloud_DA_forward_operator.ceilometer_hofx_driver`
- **localization**: Localization options.
  - **use**: Option to use localization. Set to True to use localization, False if localization is not desired.
  - **lh**: Horizontal localization (km).
  - **lv**: Vertical localization (model vertical levels).

## Plotting options

- **save_tag**: String added to all output file names.
- **out_dir**: Directory to save output to.
- **obs_plots**: Options for plotting observations.
  - **ceil**: Options used for plotting observed cloud ceilings. Consists of keyword arguments passed directly to matplotlib.pyplot.scatter.
- **postage_stamp_plots**: Options for postage stamp plots (i.e., each ensemble member plotted in a separate panel). Plots are 2D horizontal cross sections plotted at the vertical level closest to the average height of all assimilated observations. Includes multiple levels:
  - Variable name (Naming convention follows that used in Xarray when using pyNIO as the engine).
    - **title**: Plot title.
    - **save_tag**: String added to the output file name.
    - **cntf_kw**: Keyword arguments passed to matplotlib.pyplot.contourf.
    - **ob_plot**: Options for plotting observation locations.
      - **use**: Boolean controlling whether observation locations are included.
      - **kw**: Keyword arguments passed to matplotlib.pyplot.plot.
- **plot_postage_config**: Configuration options used for all postage stamp plots.
  - **nrows**: Number of rows.
  - **ncols**: Number of columns.
  - **figsize**: Figure size.
  - **skewt**: Option to plot skew-T, logp diagrams.
  - **lapse_rate**: Option to plot vertical profiles of lapse rates.
  - **z_max**: maximum height used for the skew-T, logp and lapse rate plots (m).
  - **pseudo_ceil_RH_thres**: Relative humidity threshold used for diagnosing pseudo cloud ceilings.
- **ens_stats_plots**: Options for plotting ensemble statistic plots. At the moment, all of these plots are 2D horizontal cross sections. Includes multiple levels:
  - Variable name (Naming convention follows that used in Xarray when using pyNIO as the engine).
    - **cntf_kw**: Keyword arguments passed to matplotlib.pyplot.contourf.
    - **ob_plot**: Options for plotting observation locations.
      - **use**: Boolean controlling whether observation locations are included.
      - **kw**: Keyword arguments passed to matplotlib.pyplot.plot.
- **plot_stat_config**: Configuration options used for all ensemble statistic plots.
  - **plot_bgd_once**: Option to plot the background only once.
  - **nrows**: Number of rows.
  - **ncols**: Number of columns.
  - **figsize**: Figure size.
  - **klvls**: Vertical levels to include.
