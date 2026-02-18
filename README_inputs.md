# Input YAML File Description

The inputs for `ceilometer_obs_enkf.py` come from a single YAML file. An example YAML file can be found here: `tests/sample.yml`. This README includes descriptions of all fields in an input YAML file.

## *Conventions*
Inputs are broken down by various sections (`ens`, `obs`, `DA`, etc.) which correspond to the outermost level of the YAML file. Within each section, inputs are given in a single table with nested inputs indicated with a `/`. A default value of "N/A" in this table either indicates that (a) there is no default value, so the parameter must be set in the YAML file or (b) that table row is purely informational, which is typically the case if there are other parameters nested underneath that parameter. As an example, consider the following YAML snippet:

```
DA:
  ob_var: 156.25
  hofx_kw:
    hgt_lim_kw:
      max_hgt: 85
    clr_ob_kw:
      clr_ob_locs: [35, 70]
```

In this case, the documentation for the `DA` section will be formatted as follows:

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| ob_var | A | $x_{1}$ |
| hofx_kw | B | N/A |
| hofx_kw/hgt_lim_kw | C | N/A | 
| hofx_kw/hgt_lim_kw/max_hgt | D | $x_{2}$ |
| hofx_kw/clr_ob_kw | E | N/A | 
| hofx_kw/clr_ob_kw/clr_ob_locs | F | $x_{3}$ |

## Ensemble Information (`ens`)

Parameters related to the ensemble

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| in_path | Template for the ensemble background files. Must include a {num} placeholder for the ensemble member number. | N/A |
| out_path | Template for the ensemble analysis files produced by the program. Must include a {num} placeholder for the ensemble member number. | N/A |
| fix_file | File containing the cell latitude and longitude information (only needed when using MPAS netCDF files). | N/A |
| type | Ensemble file type. Only fully functioning option at the moment is 'mpas'. | N/A |
| nmem | Number of ensemble members. | N/A |
| n_zlvl | Number of vertical levels starting from the surface to read from the ensemble members (omit or set to `None` to read all vertical levels). E.g., to only use the 10 lowest model levels, set `n_zlvl: 10`. | None |
| verbose | Verbosity level for ensemble I/O. Larger numbers result in more output being printed as the program runs. | N/A |

## Observation Information (`obs`)

Parameters related to the ceilometer observations

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| fname | File containing ceilometer observations. | N/A |
| domain | Only retain observations within the specified spatial domain. Must specify 4 values: [minlat, minlon, maxlat, maxlon]. Latitudes are in deg N and longitudes are in deg E. | N/A |
| entire_file | Option to assimilate all ceilometer observations from all stations within the observation file. | N/A |
| ob_sel | Specific station IDs to assimilate. Must set `entire_file: False` to use this option. Consists of a dictionary where the key is the station ID and the value is a list containing the vertical indices (starting at 0) of the observations from that station to assimilate after running the forward operator (so clear observations are included). To assimilate all observations from a single station, set the value to an empty list. | N/A |
| lim_DHR | Option to only keep observations from a single time (i.e., the DHR value closest to 0) if there are observations from multiple times from a single ceilometer. | N/A |
| verbose | Verbosity level for observation I/O. Larger numbers result in more output being printed as the program runs. | N/A |

## DA Settings (`DA`)

Parameters related to the data assimilation algorithm, including the forward operator.

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| perform_da | Option to actually run the EnSRF. If set to false, O-B values are computed, but no DA is performed. | N/A |
| skip_zero_omb | Option to skip assimilating observations where all O-B values are 0. These observations will have no impact on the analysis, so it is more efficient to skip them. This should almost always be set to True. | N/A |
| state_vars | Fields from the ensemble background files to include in the state vector used for DA. | N/A |
| ob_var | Observation error variance (in %^2). | N/A |
| | | |
| hofx_kw | Various keywords passed to the forward operator (`main.cloud_DA_forward_operator.ceilometer_hofx_driver()`). | N/A |
| hofx_kw/debug | Debug level. Increase for more debugging output. | 0 |
| hofx_kw/verbose | Verbosity level for the forward operator. Larger numbers result in more output being printed as the program runs. | 1 |
| hofx_kw/cld_field | Name of the cloud fraction field from the ensemble background files. | cldfrac |
| hofx_kw/interp_col_kw | Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.interp_model_col_to_ob()` | N/A |
| hofx_kw/interp_col_kw/method | Interpolation method. Only supported method currently is 'nearest' | nearest |
| hofx_kw/interp_col_kw/proj_str | Map projection string for pyproj. Horizontal interpolation is performed in map projection space. | '+proj=lcc +lat_0=39 +lon_0=-96 +lat_1=33 +lat_2=45' |
| hofx_kw/hgt_lim_kw | Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.impose_hgt_limits()` | N/A |
| hofx_kw/hgt_lim_kw/min_hgt | Only assimilate cloud fraction observations above this height (m AGL) | 10 |
| hofx_kw/hgt_lim_kw/max_hgt | Only assimilate cloud fraction observations below this height (m AGL) | 3658 |
| hofx_kw/min_frac_kw | Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.impose_min_cld_frac()` | N/A |
| hofx_kw/min_frac_kw/min_cld_frac | Model cloud fraction below which cloud fractions are set to 0 (%) | 5 |
| hofx_kw/clr_ob_kw | Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.add_clear_obs()` | N/A |
| hofx_kw/clr_ob_kw/clr_ob_locs | Vertical locations for clear observations (m AGL). Necessary because ceilometers only report where there are clouds. | range(250, 3500, 500) |
| hofx_kw/clr_ob_kw/terminate_clr_col | CLAM values used to terminate column of clear obs (i.e., do not add clear obs above the height of one of the CLAM values)| [8] |
| hofx_kw/interp_z_kw | Keyword arguments passed to `main.cloud_DA_forward_operator.sfc_cld_forward_operator.interp_model_to_obs()` | N/A |
| hofx_kw/interp_z_kw/method | Interpolation method. Passed to si.interp1d | nearest |
| hofx_kw/interp_z_kw/match_precision | Option to set the interpolated cloud amount from the model to the same cloud amount from the obs if: (1) the model cloud amount is within margin of error for the observation (usually 12.5 or 25%) and (2) both the model and observed cloud amounts are between 1 and 99%. | True |
| | | |
| localization | Keyword arguments used for localization. The fifth-order function from Gaspari and Cohn (1999, QJRMS), their eqn (4.10), is used for localization. | N/A |
| localization/use | Option to use localization. | N/A |
| localization/lh | Horizontal localization half length (km). The localization function goes to 0 at 2lh. | N/A |
| localization/lv | Vertical localization half length (vertical model levels). The localization function goes to 0 at 2lv. | N/A |
| | | |
| verbose | Verbosity level for running the EnSRF. Larger numbers result in more output being printed as the program runs. | N/A |
| diag_file | File to write DA diagnostic output to. | N/A |
| update_hofx_with_enkf | Option to update H(x) after each observation is assimilated using a separate EnKF call. | N/A |
