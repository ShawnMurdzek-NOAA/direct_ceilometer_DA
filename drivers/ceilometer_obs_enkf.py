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
import direct_ceilometer_DA.main.cloud_DA_enkf_postprocess as ens_post
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


def extract_ceil_obs(ens_obj):
    """
    Extract ceilometer observations used for DA

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    
    Returns
    -------
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator

    """

    bufr_df = ens_obj._subset_bufr(['ADPSFC', 'MSONET'], DHR=np.nan)
    cld_ob_df = cfo.remove_missing_cld_ob(bufr_df)

    return cld_ob_df


def subset_ceil_obs(cld_ob_df, param, da_exp):
    """
    Only retain the ceilometer observations desired for a certain experiment

    Parameters
    ----------
    cld_ob_df : pd.DataFrame
        Full set of ceilometer observations
    param : dictionary
        Input parameters
    da_exp : string
        DA experiment. Key within the "ob_sel" part of param
    
    Returns
    -------
    subset_ob_df : pd.DataFrame
        Ceilometer observations used in this particular experiment

    """

    # Determine SIDs to include when computing the forward operator
    # Use all available SIDs if experiment is set to "entire_file"
    if param['ob_sel'][da_exp] == 'entire_file':
        subset_ob_df = copy.deepcopy(cld_ob_df)
    else:
        SIDs = list(param['ob_sel'][da_exp].keys())
        cond = np.zeros(len(cld_ob_df))
        for s in SIDs:
            cond = cond + (cld_ob_df['SID'] == s)
        subset_ob_df = copy.deepcopy(cld_ob_df.loc[cond > 0, :])

    return subset_ob_df


def read_preprocess_ens_obs(yml_fname):
    """
    Read input YAML file, then read in and preprocess ensemble output and BUFR obs

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
    zlvls : array
        Index corresponding to each model level
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
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

    # Subset BUFR ceilometer obs
    cld_ob_df = extract_ceil_obs(ens_obj)

    return ens_obj, z1d, zlvls, cld_ob_df, param


def run_cld_forward_operator(ens_obj, cld_ob_df, ens_name=['mem0001'], hofx_kw={}, verbose=False):
    """
    Run the cloud DA forward operator for all observations in the subset domain

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    ens_name : list of strings, optional
        Ensemble names
    hofx_kw : dictionary, optional
        Keyword arguments passed to cfo.ceilometer_hofx_driver()
    verbose : boolean, optional
        Option to print extra output
    
    Returns
    -------
    cld_hofx : dictionary of cfo.sfc_cld_forward_operator objects
        Ceilometer forward operator output for each ensemble member

    """
    
    cld_hofx = {}

    # Run forward operator
    for n in ens_name:
        if verbose: print(f'Running forward operator on ensemble member {n}')
        model_ds = ens_obj.subset_ds[n]
        cld_hofx[n] = cfo.ceilometer_hofx_driver(cld_ob_df, model_ds, **hofx_kw)
    
    return cld_hofx


def ceil_ob_locs(ens_obj, cld_ob_df):
    """
    Determine ceilometer observation locations in (z model, lon, lat)

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    
    Returns
    -------
    cld_coord_model : list
        Observation location in model coordinates (z, lon, lat)

    """

    cld_coord_model = [0, cld_ob_df['XOB'].values[0] - 360, cld_ob_df['YOB'].values[0]]
    cld_hofx = run_cld_forward_operator(ens_obj, cld_ob_df, ens_name=[ens_obj.mem_names[0]])
    

    return cld_coord_model


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


def run_enkf(ens_obj, ob_df, param, verbose=0):
    """
    Run EnKF for an arbitrary number of observations

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    ob_df : pd.DataFrame
        Ceilometer observations
    param : dictionary
        Input YAML parameters'
    verbose : int, optional
        Verbosity level, by default 1

    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    cld_ob_coord : list
        Observed cloud coordinates in model space. Dimensions: (z, lon, lat)
        
    """

    start_enkf = dt.datetime.now()
    cld_ob_coord = []

    # Apply cloud DA forward operator on first ensemble member to get locations of clear obs
    m1 = ens_obj.mem_names[0]
    cld_hofx_ref = run_cld_forward_operator(ens_obj, ob_df, ens_name=[m1], hofx_kw=param['hofx_kw'])

    # Apply cloud DA forward operator if only needed once
    if (not param['redo_hofx']) or (not param['perform_da']):
        cld_hofx = run_cld_forward_operator(ens_obj, ob_df, ens_name=ens_obj.mem_names, hofx_kw=param['hofx_kw'])
        if verbose > 0: print(f"Time to complete forward operator for all members and obs = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    # Loop over each observation
    if param['ob_sel'][da_exp] == 'entire_file':
        ob_sids = cld_hofx_ref[m1].data['SID']
    else:
        ob_sids = list(param['ob_sel'][da_exp].keys())
    for i, s in enumerate(ob_sids):
        if param['ob_sel'][da_exp] == 'entire_file':
            ob_idx = list(range(len(cld_hofx_ref[m1].data['HOCB'][i])))
        else:
            ob_idx = param['ob_sel'][da_exp][s]
        for j in ob_idx:
            start_loop = dt.datetime.now()
            if verbose > 1: print(f"  Looping over ob {s} {j}")

            # Run forward operator
            if (param['redo_hofx']) and (param['perform_da']):
                dum = ob_df.loc[ob_df['SID'] == s, :]
                cld_hofx = run_cld_forward_operator(ens_obj, dum, ens_name=ens_obj.mem_names, hofx_kw=param['hofx_kw'])
                idx1 = 0
            else:
                idx1 = np.where(np.array(cld_hofx[m1].data['SID']) == s)[0][0]

            # Extract cloud amount, H(x), and location
            hofx = np.zeros(len(cld_hofx))
            cld_ob_coord.append([0, cld_hofx[m1].data['lon'][idx1], cld_hofx[m1].data['lat'][idx1]])
            for k, mem in enumerate(ens_obj.mem_names):
                hofx[k] = cld_hofx[mem].data['hofx'][idx1][j]
                cld_ob_coord[-1][0] = cld_ob_coord[-1][0] + cld_hofx[mem].data['ob_hgt_model'][idx1][j]
            cld_ob_coord[-1][0] = cld_ob_coord[-1][0] / len(ens_obj.mem_names)
            cld_amt = cld_hofx[mem].data['ob_cld_amt'][idx1][j]

            # Skip remaining steps if not performing DA
            if not param['perform_da']:
                continue
            
            # Compute localization
            if param['localization']['use']:
                start_local = dt.datetime.now()
                if verbose > 2: print(f"  computing localization with lh = {param['localization']['lh']}, lv = {param['localization']['lv']}")
                C_local = compute_localization_array(ens_obj, param, cld_ob_coord[-1][0], cld_ob_coord[-1][1], cld_ob_coord[-1][2])
                if verbose > 0: print(f"  Time to complete localization = {(dt.datetime.now() - start_local).total_seconds()} s")
            else:
                C_local = None

            # Run EnKF
            enkf_obj = enkf.enkf_1ob(ens_obj.state_matrix['data'], cld_amt, hofx, param['ob_var'], localize=C_local)
            enkf_obj.EnSRF()

            # Update ens_obj with the new analysis
            xa_nd = unravel_state_matrix(enkf_obj.x_a, ens_obj)
            for v in xa_nd.keys():
                for ens in xa_nd[v].keys():
                    ens_obj.subset_ds[ens][v].values = xa_nd[v][ens]
            ens_obj.state_matrix['data'] = enkf_obj.x_a

            if verbose > 0: print(f"  Time to assimilate {s} {j} = {(dt.datetime.now() - start_loop).total_seconds()} s")

    if verbose > 0: print(f"run_enkf.py total time = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    return ens_obj, np.array(cld_ob_coord)


def only_retain_bgd_stats(param):
    """
    Only retain background stats plots in param

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    param : dictionary
        Input parameters, but only with background stats plots included

    """

    # Turn off all other plots
    param['postage_stamp_plots'] = {}
    param['plot_postage_config']['skewt'] = False

    # Only retain bgd ensemble stat plots
    keep_dict = {}
    for key in param['ens_stats_plots']:
        if 'bgd' in key:
            keep_dict[key] = param['ens_stats_plots'][key]
    param['ens_stats_plots'] = keep_dict

    # Turn off option to ignore bgd ensemble stat plots
    param['plot_stat_config']['plot_bgd_once'] = False

    return param


if __name__ == '__main__':

    start = dt.datetime.now()

    # Read and preprocess ensemble
    ens_obj, ens_z1d, ens_zlvls, cld_ob_df, param = read_preprocess_ens_obs(sys.argv[1])
    ens_obj_original = copy.deepcopy(ens_obj)
    param_original = copy.deepcopy(param)

    # Create plot of observations
    print('create plot with obs cloud fractions')
    bins = [0] + list(0.5*(ens_z1d[param['plot_stat_config']['klvls']][1:] + 
                           ens_z1d[param['plot_stat_config']['klvls']][:-1]))
    fig = ens_viz.plot_cld_obs(ens_obj, cld_ob_df, param, bins=bins, 
                               nrows=param['plot_stat_config']['nrows'], 
                               ncols=param['plot_stat_config']['ncols'],
                               scatter_kw={'vmin':0, 'vmax':100, 'cmap':'plasma_r', 's':32, 'edgecolors':'k', 'linewidths':0.5})
    plt.savefig(f"{param['out_dir']}/obs_clouds.png", dpi=500)
    plt.close(fig)

    # Loop over each experiment
    for da_exp in param['ob_sel'].keys():

        # Start with fresh versions of param and ens_obj
        ens_obj = copy.deepcopy(ens_obj_original)
        param = copy.deepcopy(param_original)

        start_loop = dt.datetime.now()
        print()
        print('-----------------------------------------------')
        print(f"Starting EnKF experiment {da_exp}")

        # Extract only the stations needed for this experiment
        subset_ob_df = subset_ceil_obs(cld_ob_df, param, da_exp)

        # Run EnKF
        ens_obj, ob_coord_all = run_enkf(ens_obj, subset_ob_df, param, verbose=2)
        ens_obj = ens_post.post_enkf(ens_obj, param, DA=param['perform_da'])

        # Make plots
        print()
        print('Plotting section')
        save_dir = f"{param['out_dir']}/{da_exp}"
        os.system(f"mkdir {save_dir}")
        ens_viz.plot_driver(ens_obj, param, save_dir, ob_coord_all, ens_zlvls, ens_z1d, verbose=1)
        
        print(f'total time for {da_exp} = {(dt.datetime.now() - start_loop).total_seconds()} s')

    # Plot background ensemble stat fields if only plotting once
    if param['plot_stat_config']['plot_bgd_once']:
        param = only_retain_bgd_stats(param)
        ob_coord_all = np.array([[0, 0, 0]])
        print()
        print('Final plots: Background ensemble stats')
        save_dir = f"{param['out_dir']}"
        ens_viz.plot_driver(ens_obj, param, save_dir, ob_coord_all, ens_zlvls, ens_z1d, verbose=1)
    
    print()
    print(f'total elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End ceilometer_obs_enkf.py
"""