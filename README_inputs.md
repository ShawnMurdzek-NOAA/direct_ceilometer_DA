# Input YAML File Description

The inputs for `ceilometer_obs_enkf.py` come from a single YAML file. An example YAML file can be found in `tests/sample.yml`. This README includes descriptions of the various fields in an input YAML file.

## Ensemble Information
- **in_path**: Template for the ensemble background files. Must include a {num} placeholder for the ensemble member number.
- **out_path**: Template for the ensemble analysis files produced by the program. Must include a {num} placeholder for the ensemble member number.
- **fix_file**: File containing the cell latitude and longitude information (only needed when using MPAS netCDF files).
- **type**: Ensemble file type. Only fully functioning option at the moment is 'mpas'.
- **nmem**: Number of ensemble members.
- **n_zlvl**: Number of vertical levels starting from the surface to read from the ensemble members (omit to read all vertical levels). E.g., to only use the 10 lowest model levels, set `n_zlvl: 10`.
- **verbose**: Verbosity level for ensemble I/O. Larger numbers result in more output being printed as the program runs.

## Observation Information
- **fname**: File containing ceilometer observations.
- **domain**: Only retain observations within the specified spatial domain. Must specify 4 values: [minlat, minlon, maxlat, maxlon]. Latitudes are in deg N and longitudes are in deg E.
- **entire_file**: Option to assimilate all ceilometer observations from all stations within the observation file.
- **ob_sel**: Specific station IDs to assimilate. Must set `entire_file: False` to use this option. Consists of a dictionary where the key is the station ID and the value is a list containing the vertical indices (starting at 0) of the observations from that station to assimilate after running the forward operator (so clear observations are included). To assimilate all observations from a single station, set the value to an empty list.
- **lim_DHR**: Option to only keep observations from a single time (i.e., the DHR value closest to 0) if there are observations from multiple times from a single ceilometer.
- **verbose**: Verbosity level for observation I/O. Larger numbers result in more output being printed as the program runs.

## DA Settings
- **perform_da**: Option to actually run the EnSRF. If set to false, O-B values are computed, but no DA is performed.
- **skip_zero_omb**: Option to skip assimilating observations where all O-B values are 0. These observations will have no impact on the analysis, so it is more efficient to skip them. This should almost always be set to True.
- **state_vars**: Fields from the ensemble background files to include in the state vector used for DA.
- **ob_var**: Observation error variance (in %^2).
- **hofx_kw**: Various keywords passed to the forward operator (`main.cloud_DA_forward_operator.ceilometer_hofx_driver()`). Additional options and defaults values can be found in `main/cloud_DA_forward_operator.py`.
  - **hgt_lim_kw**: Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.impose_hgt_limits()`
    - **max_hgt**: Only assimilate cloud fraction observations below this height (m AGL)
  - **clr_ob_kw**: Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.add_clear_obs()`
    - **clr_ob_locs**: Vertical locations for clear observations (m AGL). Necessary because ceilometers only report where there are clouds.
  - **cld_field**: Name of the cloud fraction field from the ensemble background files.
  - **verbose**: Verbosity level for the forward operator. Larger numbers result in more output being printed as the program runs.
- **localization**: Keyword arguments used for localization. The fifth-order function from Gaspari and Cohn (1999, QJRMS), their eqn (4.10), is used for localization.
  - **use**: Option to use localization.
  - **lh**: Horizontal localization half length (km). The localization function goes to 0 at 2lh.
  - **lv**: Vertical localization half length (vertical model levels). The localization function goes to 0 at 2lv.
- **verbose**: Verbosity level for running the EnSRF. Larger numbers result in more output being printed as the program runs.
- **diag_file**: File to write DA diagnostic output to.
- **update_hofx_with_enkf**: Option to update H(x) after each observation is assimilated using a separate EnKF call.
