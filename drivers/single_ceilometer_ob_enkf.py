"""
Ceilometer Observation DA Tests Using an EnKF

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import metpy.calc as mc
from metpy.units import units
import copy
import yaml

import probing_rrfs_ensemble as pre
import direct_ceilometer_DA.main.cloud_DA_forward_operator as cfo
import direct_ceilometer_DA.main.cloud_DA_enkf_viz as ens_viz
from pyDA_utils import enkf
import pyDA_utils.ensemble_utils as eu
from pyDA_utils import bufr
import pyDA_utils.colormaps as cm
import pyDA_utils.localization as local


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def ens_avg_z_1d(ens_obj):
    """
    Return a 1D array of the average height AGL for each model level

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    
    Returns
    -------
    z1d : array
        Average height AGL for each model level
    zlvls : array
        Index corresponding to each model level

    """

    mem_name = ens_obj.mem_names[0]
    z1d = np.mean(ens_obj.subset_ds[mem_name]['HGT_P0_L105_GLC0'].values - 
                  ens_obj.subset_ds[mem_name]['HGT_P0_L1_GLC0'].values[np.newaxis, :, :], axis=(1, 2))
    zlvls = ens_obj.subset_ds[mem_name]['lv_HYBL2'].values

    return z1d, zlvls


def read_ens_from_nc(param):
    """
    Read in an ensemble subset from a netCDF file

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output

    """

    ens_obj = eu.read_subset_ens_nc(param['subset_ens_nc'])
    ens_obj.verif_obs = bufr.bufrCSV(param['bufr_fname'])
    ens_obj.state_matrix = ens_obj._create_state_matrix(param['state_vars'])
    ens_obj.state_matrix.update(ens_obj._compute_ens_stats())
    ens_obj.state_matrix['ens_dev'] = ens_obj._compute_ens_deviations()

    return ens_obj


def reformat_param_plot_opts(param):
    """
    Reformat plotting options in input parameters.

    Reformatting includes adding custom colormaps and constructing contour intervals

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    param : dictionary
        Input parameters

    """

    cmap_dict = cm.generate_cust_cmaps_dict()
    for key in ['postage_stamp_plots', 'ens_stats_plots']:
        for field in param[key].keys():

            # Expand clevels, if needed
            clevels = param[key][field]['cntf_kw']['levels']
            if type(clevels) is dict:
                clevels_range = np.arange(clevels['range'][0], clevels['range'][1], clevels['range'][2])
                param[key][field]['cntf_kw']['levels'] = clevels_range
            
            # Use custom colormap dictionary
            cmap_name = param[key][field]['cntf_kw']['cmap']
            if type(cmap_name) is str:
                param[key][field]['cntf_kw']['cmap'] = cmap_dict[cmap_name]
    
    return param


def read_preprocess_ens(yml_fname):
    """
    Read input YAML file, then read in and preprocess ensemble output

    Parameters
    ----------
    yml_fname : string
        Input YAML file name
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    z1d : array
        1D array of average heights AGL for each model level
    param : dictionary
        Input parameters

    Notes
    -----
    This step is independent of the observation being assimilated, so it should only need to be done
    once

    """

    # Read input parameters
    with open(yml_fname, 'r') as fptr:
        param = yaml.safe_load(fptr)
    param = reformat_param_plot_opts(param)

    # Read in ensemble output
    try:
        ens_obj = read_ens_from_nc(param)
        print('using subset ensemble output from netCDF file')
    except:
        ens_obj = pre.read_ensemble_output(param)
        if param['save_to_nc']:
            ens_obj.save_subset_ens(param['subset_ens_nc'])
    
    # Save background fields
    for ens in ens_obj.mem_names:
        for v in param['state_vars']:
            ens_obj.subset_ds[ens]['bgd_'+v] = ens_obj.subset_ds[ens][v].copy()

    # Create 1D array of average heights AGL
    z1d, zlvls = ens_avg_z_1d(ens_obj)

    return ens_obj, z1d, zlvls, param


def run_cld_forward_operator_1ob(ens_obj, ob_sid, ob_idx, ens_name=['mem0001'], hofx_kw={}, verbose=False):
    """
    Run the cloud DA forward operator for a single observation

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    ob_sid : string
        Observation station ID
    ob_idx : integer
        Observation index. Each ob_sid may have multiple obs (especially after the forward operator
        adds clear obs), so ob_idx selects which observation to use from a particular station.
    ens_name : list of strings, optional
        Ensemble names
    hofx_kw : dictionary, optional
        Keyword arguments passed to cfo.ceilometer_hofx_driver()
    verbose : boolean, optional
        Option to print extra output
    
    Returns
    -------
    cld_amt : float
        Observed cloud amount (%)
    cld_z : float
        Observed cloud base (m AGL)
    hofx_output : list
        Model cloud amount from each ensemble member (%). Height is the same as cld_z.
    cld_coord_model : list
        Observation location in model coordinates (z, lon, lat)

    """
    
    hofx_output = np.zeros(len(ens_name))

    # Select observation for DA
    bufr_df = ens_obj._subset_bufr(['ADPSFC', 'MSONET'], DHR=np.nan)
    dum = bufr_df.loc[bufr_df['SID'] == ob_sid, :]
    cld_ob_df = cfo.remove_missing_cld_ob(dum)
    cld_coord_model = [0, cld_ob_df['XOB'].values[0] - 360, cld_ob_df['YOB'].values[0]]
    if verbose: print("Observation:\n", cld_ob_df.loc[:, ['TYP', 'SID', 'XOB', 'YOB', 'CLAM', 'HOCB']])
    
    # Run forward operator
    for i, n in enumerate(ens_name):
        if verbose: print(f'Running forward operator on ensemble member {n}')
        model_ds = ens_obj.subset_ds[n]

        # Create ceilometer forward operator object
        cld_hofx = cfo.ceilometer_hofx_driver(cld_ob_df, model_ds, **hofx_kw)
        if verbose: 
            print("cld_hofx.data['CLAM'] = ", cld_hofx.data['CLAM'])
            print("cld_hofx.data['HOCB'] = ", cld_hofx.data['HOCB'])
            print("cld_hofx.data['ob_cld_amt'] = ", cld_hofx.data['ob_cld_amt'])
            print("cld_hofx.data['hofx'] = ", cld_hofx.data['hofx'])
        hofx_output[i] = cld_hofx.data['hofx'][0][ob_idx]
        cld_amt = cld_hofx.data['ob_cld_amt'][0][ob_idx]
        cld_z = cld_hofx.data['HOCB'][0][ob_idx]
        cld_coord_model[0] = cld_coord_model[0] + cld_hofx.data['ob_hgt_model'][0][ob_idx]
    cld_coord_model[0] = cld_coord_model[0] / len(ens_name)
    
    return cld_amt, cld_z, hofx_output, cld_coord_model


def compute_localization_array(ens_obj, param, z, lon, lat):
    """
    Compute localization array for EnKF DA

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    param : dictionary
        YAML inputs
    z : float
        Observation height
    lon : float
        Observation longitude (deg E)
    lat : float
        Observation latitude (deg N)
    
    Returns
    -------
    C : np.ndarray
        Localization array

    """
  
    # Use Gaspari and Cohn (1999) 5th-order localization fct
    local_fct = local.localization_fct(local.gaspari_cohn_5ord)

    # Extract information needed to compute localization
    model_pts = ens_obj.state_matrix['loc']
    ob_pt = np.array([z, lat, lon])
    lh = param['localization']['lh']
    lv = param['localization']['lv']

    # Compute localization
    C = local_fct.compute_localization(model_pts, ob_pt, lv, lh)

    return C


def unravel_state_matrix(x, ens_obj, ens_dim=True):
    """
    Unravel state matrix from ens_obj so fields can be plotted

    Parameters
    ----------
    x : array
        State matrix. Dimensions (M, N), where M is the (number of gridpoints) X (number of fields)
        and N is the number of ensemble members
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    ens_dim : boolean, optional
        Option to also unravel the ensemble dimension. Set to False if x is 1D
    
    Returns
    -------
    output : dictionary
        Unraveled state matrix. Keys are the different fields, and each field is now 3D
    
    """

    output = {}
    for v in np.unique(ens_obj.state_matrix['vars']):
        var_cond = ens_obj.state_matrix['vars'] == v
        if ens_dim:
            output[v] = {}
            for i, ens in enumerate(ens_obj.mem_names):
                output[v][ens] = np.reshape(x[var_cond, i], ens_obj.subset_ds[ens][v].shape)
        else:
            output[v] = np.reshape(x[var_cond], ens_obj.subset_ds[ens_obj.mem_names[0]][v].shape)

    return output


def compute_RH(ens_obj, prefix=''):
    """
    Compute RH for the ensemble subset spatial domain

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    prefix : string, optional
        RH field is written to "{prefix}RH_P0_L105_GLC0" using "PRES_P0_L105_GLC0", 
        "{prefix}TMP_P0_L105_GLC0", and "{prefix}SPFH_P0_L105_GLC0"
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with the added RH fields

    """

    for mem in ens_obj.mem_names:
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"] = ens_obj.subset_ds[mem]['SPFH_P0_L105_GLC0'].copy()
        p = ens_obj.subset_ds[mem]["PRES_P0_L105_GLC0"].values * units.Pa
        T = ens_obj.subset_ds[mem][f"{prefix}TMP_P0_L105_GLC0"].values * units.K
        Q = ens_obj.subset_ds[mem][f"{prefix}SPFH_P0_L105_GLC0"].values * units.kg / units.kg
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].values = mc.relative_humidity_from_specific_humidity(p, T, Q).magnitude * 100
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].attrs['long_name'] = 'relative humidity'
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].attrs['units'] = '%'

    return ens_obj


def compute_ens_stats_3D(ens_obj, field, stat_fct={'mean_':np.mean, 'std_': np.std}):
    """
    Compute various ensemble stats for a 3D field that is not part of the state vector

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    field : string
        Field from ens_obj.subset_ds[m] to compute statistics for
    stat_fct : dictionary, optional
        Statistics to compute. Key is the prefix added to the resulting field and the value is the 
        function
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with statistics for the desired field added to the first ensemble member

    Notes
    -----
    Output is saved to the first ensemble member

    """

    mem_names = ens_obj.mem_names
    shape = ens_obj.subset_ds[mem_names[0]][field].shape
    full_array = np.zeros([shape[0], shape[1], shape[2], len(mem_names)])

    # Extract the desired field from each ensemble member
    for i, m in enumerate(mem_names):
        full_array[:, :, :, i] = ens_obj.subset_ds[m][field].values
    
    # Compute stats
    for f in stat_fct.keys():
        ens_obj.subset_ds[mem_names[0]][f"{f}{field}"] = ens_obj.subset_ds[mem_names[0]][field].copy()
        ens_obj.subset_ds[mem_names[0]][f"{f}{field}"].values = stat_fct[f](full_array, axis=3)

    return ens_obj


def compute_ens_incr_3D(ens_obj, field):
    """
    Compute analysis increments for the desired field

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    field : string
        Field from ens_obj.subset_ds[m] to compute statistics for
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with analysis increments (using the prefix "incr_")

    Notes
    -----
    The mean analysis increment is saved to the first ensemble member with the prefix "mean_incr_"

    """

    mem_names = ens_obj.mem_names
    incr_sum = np.zeros(ens_obj.subset_ds[mem_names[0]][f"bgd_{field}"].shape)

    # Compute increment for each ensemble member
    for m in mem_names:
        ens_obj.subset_ds[m][f"incr_{field}"] = ens_obj.subset_ds[m][f"bgd_{field}"].copy()
        ens_obj.subset_ds[m][f"incr_{field}"].values = ens_obj.subset_ds[m][f"ana_{field}"].values - ens_obj.subset_ds[m][f"bgd_{field}"].values
        incr_sum = incr_sum + ens_obj.subset_ds[m][f"incr_{field}"].values
    
    # Add average increment to first ensemble member
    ens_obj.subset_ds[mem_names[0]][f"mean_incr_{field}"] = ens_obj.subset_ds[mem_names[0]][f"bgd_{field}"].copy()
    ens_obj.subset_ds[mem_names[0]][f"mean_incr_{field}"].values = incr_sum / len(mem_names)

    return ens_obj


def add_inc_and_analysis_to_ens_obj(ens_obj, enkf_obj):
    """
    Add the analysis increment and analysis fields to ens_obj for easier plotting

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    enkf_obj : pyDA_utils.enkf.enkf_1ob object
        EnKF output
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with the analysis increment and analysis fields added

    """

    # Compute increment
    inc_1d = enkf_obj.x_a - enkf_obj.x_b

    # Turn 1D fields into nD fields
    inc_nd = unravel_state_matrix(inc_1d, ens_obj)
    xa_nd = unravel_state_matrix(enkf_obj.x_a, ens_obj)

    # Add fields to ens_obj
    for out, label in zip([inc_nd, xa_nd], ['incr_', 'ana_']):
        for v in out.keys():
            for ens in out[v].keys():
                ens_obj.subset_ds[ens][label+v] = ens_obj.subset_ds[ens][v].copy()
                ens_obj.subset_ds[ens][label+v].values = out[v][ens]

    return ens_obj


def add_ens_mean_std_K_to_ens_obj(ens_obj, enkf_obj):
    """
    Add the ensemble mean, standard deviation, and Kalman gain to the first ensemble member

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    enkf_obj : pyDA_utils.enkf.enkf_1ob object
        EnKF output
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with the ensemble mean, standard deviation, and K added to the first
        ensemble member

    """

    # Compute stats
    var_2d = {}
    for x, label1 in zip([enkf_obj.x_b, enkf_obj.x_a], ['', 'ana']):
        for fct, label2 in zip([np.mean, np.std], ['mean', 'std']):
            if label1 == '':
                name = label2
            else:
                name = f'{label2}_{label1}'
            var_2d[name] = unravel_state_matrix(fct(x, axis=1), ens_obj, ens_dim=False)
    var_2d['K'] = unravel_state_matrix(enkf_obj.K, ens_obj, ens_dim=False)

    # Add fields to ens_obj
    ens = ens_obj.mem_names[0]
    for v in var_2d[list(var_2d.keys())[0]]:
        for key in var_2d.keys():
            ens_obj.subset_ds[ens][f"{key}_{v}"] = ens_obj.subset_ds[ens][v].copy()
            ens_obj.subset_ds[ens][f"{key}_{v}"].values = var_2d[key][v]
            if key == 'K':
                units = ens_obj.subset_ds[ens][f"{key}_{v}"].attrs['units']
                ens_obj.subset_ds[ens][f"{key}_{v}"].attrs['units'] = f"{units} / [obs units]"
                ens_obj.subset_ds[ens][f"{key}_{v}"].attrs['long_name'] = "Kalman gain"
        ens_obj.subset_ds[ens][f"mean_incr_{v}"] = ens_obj.subset_ds[ens][v].copy()
        ens_obj.subset_ds[ens][f"mean_incr_{v}"].values = var_2d['mean_ana'][v] - var_2d['mean'][v]

    return ens_obj


def run_enkf_1ob(ens_obj, ob_sid, ob_idx, verbose=0):
    """
    Run EnKF for a single observation

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    ob_sid : string
        Observation SID
    ob_idx : integer
        Index corresponding to the observation being assimilated
    verbose : int, optional
        Verbosity level, by default 1

    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    cld_ob_coord : list
        Observed cloud coordinates in model space. Dimensions: (z, lon, lat)
    cld_z : float
        Observed cloud base (m AGL)
        
    """

    start_enkf = dt.datetime.now()

    # Apply cloud DA forward operator
    cld_amt, cld_z, hofx, cld_ob_coord = run_cld_forward_operator_1ob(ens_obj, ob_sid, ob_idx, 
                                                                      ens_name=ens_obj.mem_names,
                                                                      hofx_kw={'hgt_lim_kw':{'max_hgt':3500},
                                                                               'verbose':0},
                                                                      verbose=False)
    if verbose > 1: print('Cloud ceilometer ob hgt =', cld_z)
    if verbose > 1: print('Cloud ceilometer ob amt =', cld_amt)
    if verbose > 1: print('Cloud ceilometer H(x) =', hofx)
    if verbose > 0: print(f"Time to complete forward operator = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    # Compute localization
    if param['localization']['use']:
        start_local = dt.datetime.now()
        if verbose > 1: print(f"computing localization with lh = {param['localization']['lh']}, lv = {param['localization']['lv']}")
        C_local = compute_localization_array(ens_obj, param, cld_ob_coord[0], cld_ob_coord[1], cld_ob_coord[2])
        if verbose > 0: print(f"Time to complete localization = {(dt.datetime.now() - start_local).total_seconds()} s")
    else:
        C_local = None

    # Run EnKF
    enkf_obj = enkf.enkf_1ob(ens_obj.state_matrix['data'], cld_amt, hofx, param['ob_var'], localize=C_local)
    enkf_obj.EnSRF()
    if verbose > 0: print(f"Time to complete forward operator and EnSRF = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    # Update ens_obj with the new analysis
    xa_nd = unravel_state_matrix(enkf_obj.x_a, ens_obj)
    for v in xa_nd.keys():
        for ens in xa_nd[v].keys():
            ens_obj.subset_ds[ens][v].values = xa_nd[v][ens]
    ens_obj.state_matrix['data'] = enkf_obj.x_a

    return ens_obj, cld_ob_coord, cld_z


def post_enkf(ens_obj, param):
    """
    Postprocess EnKF output

    * Compute RH
    * Create analysis (ana_) fields
    * Compute ensemble statistics
    * Compute analysis increments

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    param : dictionary
        YAML inputs

    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output

    """

    # Save analysis fields
    for ens in ens_obj.mem_names:
        for v in param['state_vars']:
            ens_obj.subset_ds[ens]['ana_'+v] = ens_obj.subset_ds[ens][v].copy()

    # Compute RH
    for p in ['bgd_', 'ana_']:
        ens_obj = compute_RH(ens_obj, prefix=p)
    
    # Compute ensemble stats and analysis increment
    param['state_vars'].append('RH_P0_L105_GLC0')
    for v in param['state_vars']:
        for p in ['bgd_', 'ana_']:
            ens_obj = compute_ens_stats_3D(ens_obj, f"{p}{v}")
        ens_obj = compute_ens_incr_3D(ens_obj, v)
    
    return ens_obj


if __name__ == '__main__':

    start = dt.datetime.now()

    # Read and preprocess ensemble
    ens_obj, ens_z1d, ens_zlvls, param = read_preprocess_ens(sys.argv[1])
    ens_obj_original = copy.deepcopy(ens_obj)
    param_original = copy.deepcopy(param)

    # Create plot of observations
    print('create plot with obs cloud fractions')
    bins = [0] + list(0.5*(ens_z1d[param['plot_stat_config']['klvls']][1:] + 
                           ens_z1d[param['plot_stat_config']['klvls']][:-1]))
    fig = ens_viz.plot_cld_obs(ens_obj, param, bins=bins, 
                               nrows=param['plot_stat_config']['nrows'], ncols=param['plot_stat_config']['ncols'],
                               scatter_kw={'vmin':0, 'vmax':100, 'cmap':'plasma_r', 's':32, 'edgecolors':'k', 'linewidths':0.5})
    plt.savefig(f"{param['out_dir']}/obs_clouds.png")
    plt.close(fig)

    # Loop over each experiment
    for da_exp in param['ob_sel'].keys():

        start_loop = dt.datetime.now()
        print()
        print('-----------------------------------------------')
        print(f"Starting EnKF experiment {da_exp}")

        # Run EnKF
        ob_coord_all = []
        for ob_sid in param['ob_sel'][da_exp].keys():
            for ob_idx in param['ob_sel'][da_exp][ob_sid]:
                print(f"Assimilating {ob_sid} {ob_idx}")
                ens_obj, cld_ob_coord, cld_z = run_enkf_1ob(ens_obj, ob_sid, ob_idx, verbose=2)
                ob_coord_all.append(cld_ob_coord)
        ob_coord_all = np.array(ob_coord_all)

        # Post-EnKF
        ens_obj = post_enkf(ens_obj, param)

        # Make plots
        print()
        print('Plotting section')
        save_dir = f"{param['out_dir']}/{da_exp}"
        os.system(f"mkdir {save_dir}")
        ens_viz.plot_driver(ens_obj, param, save_dir, ob_coord_all, ens_zlvls, ens_z1d, verbose=1)
        
        # Clean up
        print(f'total time for {da_exp} (forward operator, EnSRF, and plots) = {(dt.datetime.now() - start_loop).total_seconds()} s')
        ens_obj = copy.deepcopy(ens_obj_original)
        param = copy.deepcopy(param_original)

    print()
    print(f'total elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End ceilometer_ob_enkf.py
"""