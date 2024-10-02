"""
Single Observation Ceilometer Test Using an EnSRF

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import metpy.calc as mc
from metpy.units import units
from metpy.plots import SkewT
import copy
import yaml

import probing_rrfs_ensemble as pre
import direct_ceilometer_DA.main.cloud_DA_forward_operator as cfo
from pyDA_utils import enkf
import pyDA_utils.plot_model_data as pmd
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

    """

    mem_name = ens_obj.mem_names[0]
    z1d = np.mean(ens_obj.subset_ds[mem_name]['HGT_P0_L105_GLC0'].values - 
                  ens_obj.subset_ds[mem_name]['HGT_P0_L1_GLC0'].values[np.newaxis, :, :], axis=(1, 2))

    return z1d


def read_ens_from_nc(param):
    """
    Read in an ensemble subset from a netCDF file
    """

    ens_obj = eu.read_subset_ens_nc(param['subset_ens_nc'])
    ens_obj.verif_obs = bufr.bufrCSV(param['bufr_fname'])
    ens_obj.state_matrix = ens_obj._create_state_matrix(param['state_vars'])
    ens_obj.state_matrix.update(ens_obj._compute_ens_stats())
    ens_obj.state_matrix['ens_dev'] = ens_obj._compute_ens_deviations()

    return ens_obj


def read_preprocess_ens(yml_fname):
    """
    Read in and preprocess ensemble output

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

    # Read input data
    with open(yml_fname, 'r') as fptr:
        param = yaml.safe_load(fptr)
    try:
        ens_obj = read_ens_from_nc(param)
        print('using subset ensemble output from netCDF file')
    except:
        ens_obj = pre.read_ensemble_output(param)
        if param['save_to_nc']:
            ens_obj.save_subset_ens(param['subset_ens_nc'])

    # Reformat plotting options
    cmap_dict = cm.generate_cust_cmaps_dict()
    #for key in ['postage_stamp_plots', 'ens_stats_plots']:
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

    # Create 1D array of average heights AGL
    z1d = ens_avg_z_1d(ens_obj)

    return ens_obj, z1d, param


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
    incr_sum = np.zeros(ens_obj.subset_ds[mem_names[0]][field].shape)

    # Compute increment for each ensemble member
    for m in mem_names:
        ens_obj.subset_ds[m][f"incr_{field}"] = ens_obj.subset_ds[m][field].copy()
        ens_obj.subset_ds[m][f"incr_{field}"].values = ens_obj.subset_ds[m][f"ana_{field}"].values - ens_obj.subset_ds[m][field].values
        incr_sum = incr_sum + ens_obj.subset_ds[m][f"incr_{field}"].values
    
    # Add average increment to first ensemble member
    ens_obj.subset_ds[mem_names[0]][f"mean_incr_{field}"] = ens_obj.subset_ds[mem_names[0]][field].copy()
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


def plot_horiz_slices(ds, field, ens_obj, param, verbose=False, ob={'plot':False},
                      save_dir=None):
    """
    Plot horizontal slices of the desired field at various vertical levels

    Parameters
    ----------
    ds : xr.Dataset
        Output from a single model
    field : string
        Field to plot
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    param : dictionary
        YAML inputs
    verbose : bool, optional
        Option to print extra output, by default False
    ob : dict, optional
        Observation plotting options, by default {'plot':False}
    save_dir : string, optional
        Directory to save figure to. Setting to None uses 'out_dir' from param, by default None

    Returns
    -------
    fig : plt.figure
        Figure with the desired plot

    """
    
    # Make plot
    fig = plt.figure(figsize=param['plot_stat_config']['figsize'])
    nrows = param['plot_stat_config']['nrows']
    ncols = param['plot_stat_config']['ncols']
    for i, k in enumerate(param['plot_stat_config']['klvls']):
        if verbose:
            print(f"plotting k = {k}")
        plot_obj = pmd.PlotOutput([ds], 'upp', fig, nrows, ncols, i+1)

        # Make filled contour plot
        # Skip plotting if < 2 NaN
        make_plot = np.sum(~np.isnan(ds[field][k, :, :])) > 1
        if make_plot:
            plot_obj.contourf(field, cbar=False, ingest_kw={'zind':[k]}, 
                              cntf_kw=param['ens_stats_plots'][field]['cntf_kw'])
            cax = plot_obj.cax
            meta = plot_obj.metadata['contourf0']
        else:
            if verbose:
                print(f"skipping plot for k = {k}")
            plot_obj.ax = fig.add_subplot(nrows, ncols, i+1, projection=plot_obj.proj)
        
        # Add location of observation
        if ob['plot']:
            plot_obj.plot(ob['x'], ob['y'], plt_kw=ob['kwargs'])

        plot_obj.config_ax(grid=False)
        plot_obj.set_lim(ens_obj.lat_limits[0], ens_obj.lat_limits[1], 
                         ens_obj.lon_limits[0], ens_obj.lon_limits[1])
        title = 'avg z = {z:.1f} m'.format(z=float(np.mean(ds['HGT_P0_L105_GLC0'][k, :, :] -
                                                           ds['HGT_P0_L1_GLC0'])))
        plot_obj.ax.set_title(title, size=14)
    
    plt.subplots_adjust(left=0.01, right=0.85)

    cb_ax = fig.add_axes([0.865, 0.02, 0.02, 0.9])
    cbar = plt.colorbar(cax, cax=cb_ax, orientation='vertical', aspect=35)
    cbar.set_label(f"{meta['name']} ({meta['units']})", size=14)

    plt.suptitle(field, size=18)

    # Save figure
    if save_dir is None:
        plt.savefig(f"{param['out_dir']}/{field}_{param['save_tag']}.png")
    else:
        plt.savefig(f"{save_dir}/{field}_{param['save_tag']}.png")

    return fig


def plot_horiz_postage_stamp(ens_obj, param, upp_field='TCDC_P0_L105_GLC0', klvl=0, 
                             ob={'plot':False}, save_dir=None, debug=0):
    """
    Make horizontal cross section postage stamp plots (i.e., one plot per ensemble member)

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    param : dictionary
        YAML inputs
    upp_field : string, optional
        UPP field to plot
    klvl : integer, optional
        Vertical level to plot
    ob : dict, optional
        Observation plotting options, by default {'plot':False}
    save_dir : string, optional
        Directory to save figure to. Setting to None uses 'out_dir' from param, by default None
    debug : integer, optional
        Option to print additional information for debugging. Higher numbers print more output

    Returns
    -------
    fig : plt.figure
        Plot

    """
    
    if debug > 0:
        print(param['postage_stamp_plots'][upp_field])

    # Make plot
    fig = ens_obj.postage_stamp_contourf(upp_field, 
                                         param['plot_postage_config']['nrows'], 
                                         param['plot_postage_config']['ncols'], 
                                         klvl=klvl, 
                                         figsize=param['plot_postage_config']['figsize'], 
                                         title=param['postage_stamp_plots'][upp_field]['title'],
                                         plt_kw={'ingest_kw':{'zind':[klvl]}, 
                                                 'cntf_kw':param['postage_stamp_plots'][upp_field]['cntf_kw']})
    
    # Add location of observation
    if ob['plot']:
        for ax in fig.axes:
            if type(ax) == cartopy.mpl.geoaxes.GeoAxes:
                ax.plot(ob['x'], ob['y'], transform=ccrs.PlateCarree(), **ob['kwargs'])

    # Save output
    save_fname = f"{save_dir}/postage_stamp_{param['postage_stamp_plots'][upp_field]['save_tag']}_{param['save_tag']}.png"
    plt.savefig(save_fname)

    return fig


def plot_skewt_postage_stamp(ens_obj, param, lat, lon):
    """
    Plot skew-Ts for each ensemble member in a postage stamp plot

    Parameters
    ----------

    Returns
    -------

    """

    # Create figure
    fig = plt.figure(figsize=param['plot_postage_config']['figsize'])
    nrows = param['plot_postage_config']['nrows']
    ncols = param['plot_postage_config']['ncols']

    # Loop over each ensemble member
    for i, mem in enumerate(ens_obj.mem_names):
        skew = SkewT(fig, rotation=45, subplot=[nrows, ncols, i+1])
        skew.ax.set_xlim(-40, 20)
        skew.ax.set_ylim(1000, 550)

        # Plot background and analysis
        for prefix, c, in zip(['', 'ana_'], ['b', 'r']):
            ens_obj.plot_skewts(lon, lat, fig, names=[mem], 
                                skew_kw={'hodo':False, 'barbs':False, 'skew':skew, 'bgd_lw':0.25,
                                         'Tplot_kw':{'linewidth':0.75, 'color':c}, 
                                         'TDplot_kw':{'linewidth':0.75, 'color':c},
                                         'fields':{'PRES':'PRES_P0_L105_GLC0',
                                                   'TMP':f'{prefix}TMP_P0_L105_GLC0',
                                                   'SPFH':f'{prefix}SPFH_P0_L105_GLC0'}})
        
        # Hide tick labels for certain subplots
        if (i % ncols) != 0:
            fig.axes[i].yaxis.set_ticklabels([])
        if i < (ncols * (nrows - 1)):
            fig.axes[i].xaxis.set_ticklabels([])
    
    # Add title
    plt.suptitle(f"{lat:.2f} deg N, {lon:.2f} deg E", size=16)
    
    return fig


def plot_cld_obs(ens_obj, param, bins=np.arange(0, 2001, 250), nrows=2, ncols=4, figsize=(10, 10), 
                 hofx_kw={}, scatter_kw={}):
    """
    Plot ceilometer obs in horizontal slices for various vertical bins

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    param : dictionary
        YAML inputs
    bins : array, optional
        Vertical bins used to sort ceilometer obs, by default np.arange(0, 2001, 250)
    nrows : int, optional
        Number of subplot rows, by default 2
    ncols : int, optional
        Number of subplot columns, by default 4
    figsize : tuple, optional
        Figure size, by default (10, 10)
    hofx_kw : dict, optional
        Keyword arguments passed to cfo.ceilometer_hofx_driver(), by default {}
    scatter_kw : dict, optional
        Keyword arbuments passed to plt.scatter(), by default {}

    Returns
    -------
    fig : plt.figure()
        Plot with desired figure

    """

    # Extract cloud obs
    bufr_df = ens_obj._subset_bufr(['ADPSFC', 'MSONET'], DHR=np.nan)
    cld_ob_df = cfo.remove_missing_cld_ob(bufr_df)
    cld_hofx = cfo.ceilometer_hofx_driver(cld_ob_df, ens_obj.subset_ds[ens_obj.mem_names[0]], **hofx_kw)

    # Make plot
    fig = plt.figure(figsize=figsize)
    axes = []
    for i in range(len(bins) - 1):

        # Extract obs within this height bin
        obs = {'lat':[], 'lon':[], 'ob_cld_amt':[]}
        for j in range(len(cld_hofx.data['HOCB'])):
            ob_idx = np.where(np.logical_and(cld_hofx.data['HOCB'][j] >= bins[i],
                                             cld_hofx.data['HOCB'][j] < bins[i+1]))[0]
            for k in ob_idx:
                obs['lat'].append(cld_hofx.data['lat'][j])
                obs['lon'].append(cld_hofx.data['lon'][j])
                obs['ob_cld_amt'].append(cld_hofx.data['ob_cld_amt'][j][k])

        # Make plot
        axes.append(fig.add_subplot(nrows, ncols, i+1, projection=ccrs.LambertConformal()))
        cax = axes[-1].scatter(obs['lon'], obs['lat'], c=obs['ob_cld_amt'], transform=ccrs.PlateCarree(),
                               **scatter_kw)
        axes[-1].set_extent([param['min_lon'], param['max_lon'], param['min_lat'], param['max_lat']])
        axes[-1].coastlines('50m', edgecolor='gray', linewidth=0.25)
        borders = cfeature.NaturalEarthFeature(category='cultural',
                                               scale='50m',
                                               facecolor='none',
                                               name='admin_1_states_provinces')
        axes[-1].add_feature(borders, linewidth=0.25, edgecolor='gray')
        axes[-1].set_title(f"[{bins[i]:.0f}, {bins[i+1]:.0f})", size=14)
    
    cbar = plt.colorbar(cax, ax=axes, orientation='vertical')
    cbar.set_label('observed cloud percentage', size=14)

    return fig


if __name__ == '__main__':

    start = dt.datetime.now()

    # Read and preprocess ensemble
    ens_obj, ens_z1d, param = read_preprocess_ens(sys.argv[1])
    ens_obj_original = copy.deepcopy(ens_obj)
    param_original = copy.deepcopy(param)

    # Create plot of observations
    print('create plot with obs cloud fractions')
    bins = [0] + list(0.5*(ens_z1d[param['plot_stat_config']['klvls']][1:] + 
                           ens_z1d[param['plot_stat_config']['klvls']][:-1]))
    fig = plot_cld_obs(ens_obj, param, bins=bins, 
                       nrows=param['plot_stat_config']['nrows'], ncols=param['plot_stat_config']['ncols'],
                       scatter_kw={'vmin':0, 'vmax':100, 'cmap':'plasma_r', 's':32, 'edgecolors':'k', 'linewidths':0.5})
    plt.savefig(f"{param['out_dir']}/obs_clouds.png")
    plt.close(fig)

    # Loop over each observation
    for ob_sid, ob_idx in zip(param['ob_sid'], param['ob_idx']):

        start_loop = dt.datetime.now()
        print()
        print('-----------------------------------------------')
        print(f"Starting single-ob test for {ob_sid} {ob_idx}")

        # Apply cloud DA forward operator
        cld_amt, cld_z, hofx, cld_ob_coord = run_cld_forward_operator_1ob(ens_obj, ob_sid, ob_idx, 
                                                                          ens_name=ens_obj.mem_names,
                                                                          hofx_kw={'hgt_lim_kw':{'max_hgt':3500},
                                                                                   'verbose':0},
                                                                          verbose=False)
        print('Cloud ceilometer ob hgt =', cld_z)
        print('Cloud ceilometer ob amt =', cld_amt)
        print('Cloud ceilometer H(x) =', hofx)
        print(f"Time to complete forward operator = {(dt.datetime.now() - start_loop).total_seconds()} s")

        # Compute localization
        if param['localization']['use']:
            start_local = dt.datetime.now()
            print(f"computing localization with lh = {param['localization']['lh']}, lv = {param['localization']['lv']}")
            C_local = compute_localization_array(ens_obj, param, cld_ob_coord[0], cld_ob_coord[1], cld_ob_coord[2])
            print(f"Time to complete localization = {(dt.datetime.now() - start_local).total_seconds()} s")
        else:
            C_local = None

        # Run EnKF
        enkf_obj = enkf.enkf_1ob(ens_obj.state_matrix['data'], cld_amt, hofx, param['ob_var'], localize=C_local)
        enkf_obj.EnSRF()
        print(f"Time to complete forward operator and EnSRF = {(dt.datetime.now() - start_loop).total_seconds()} s")

        # Save output to ens_obj for easier plotting
        ens_obj = add_inc_and_analysis_to_ens_obj(ens_obj, enkf_obj)
        ens_obj = add_ens_mean_std_K_to_ens_obj(ens_obj, enkf_obj)

        # Compute RH as well as ensemble stats for RH
        for p in ['', 'ana_']:
            ens_obj = compute_RH(ens_obj, prefix=p)
            ens_obj = compute_ens_stats_3D(ens_obj, f"{p}RH_P0_L105_GLC0")
        ens_obj = compute_ens_incr_3D(ens_obj, "RH_P0_L105_GLC0")
        param['state_vars'].append('RH_P0_L105_GLC0')

        # Prep for making plots
        print()
        print('Plotting section')
        save_dir = f"{param['out_dir']}/{ob_sid}_{ob_idx}"
        os.system(f"mkdir {save_dir}")

        # Make Skew-T postage stamp plots
        if param['plot_postage_config']['skewt']:
            print('plotting Skew-T diagram postage stamps...')
            fig = plot_skewt_postage_stamp(ens_obj, param, cld_ob_coord[2], cld_ob_coord[1])
            plt.savefig(f"{save_dir}/postage_stamp_skewt_{param['save_tag']}.pdf")  # Save as a PDF to make it easier to zoom in
            plt.close(fig)

        # Plot ensemble mean and standard deviation
        print('Making ensemble statistic plots')
        for field in param['ens_stats_plots']:
            if field not in ens_obj.subset_ds[ens_obj.mem_names[0]]:
                print(f'field {field} is missing. Skipping.')
                continue
            print(f'plotting {field}...')
            fig = plot_horiz_slices(ens_obj.subset_ds[ens_obj.mem_names[0]], 
                                    field,
                                    ens_obj,
                                    param,
                                    ob={'plot':True,
                                        'x':cld_ob_coord[1], 
                                        'y':cld_ob_coord[2], 
                                        'kwargs':{'marker':'*', 'color':'k'}},
                                    save_dir=save_dir)
            plt.close(fig)

        # Make postage stamp plots
        print('Making postage stamp plots')
        klvl = np.argmin(np.abs(ens_z1d - cld_z))
        print(f"postage stamp klvl = {klvl} ({ens_z1d[klvl]} m AGL)")
        for field in param['postage_stamp_plots'].keys():
            if field not in ens_obj.subset_ds[ens_obj.mem_names[0]]:
                print(f'field {field} is missing. Skipping.')
                continue
            print(f'plotting {field}...')
            fig = plot_horiz_postage_stamp(ens_obj, param, upp_field=field, 
                                           klvl=klvl,
                                           ob={'plot':True,
                                               'x':cld_ob_coord[1], 
                                               'y':cld_ob_coord[2], 
                                               'kwargs':{'marker':'*', 'color':'k'}},
                                           save_dir=save_dir,
                                           debug=0)
            plt.close(fig)
        
        # Clean up
        print(f'total time for {ob_sid} {ob_idx} (forward operator, EnSRF, and plots) = {(dt.datetime.now() - start_loop).total_seconds()} s')
        ens_obj = copy.deepcopy(ens_obj_original)
        param = copy.deepcopy(param_original)

    print(f'total elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End single_ceilometer_ob_enkf.py
"""