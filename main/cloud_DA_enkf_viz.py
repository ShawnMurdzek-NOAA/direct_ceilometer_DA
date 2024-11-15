"""
Helper functions for plotting EnSRF output fields

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt
import metpy.calc as mc
from metpy.units import units
from metpy.plots import SkewT

import direct_ceilometer_DA.main.cloud_DA_forward_operator as cfo
import pyDA_utils.plot_model_data as pmd


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def plot_horiz_slices(ds, field, ens_obj, param, verbose=False, ob_coord=[], save_dir=None):
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
    ob_coord : np.array, optional
        Observation coordinates, (lon, lat)
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
        
        # Add location of observations
        if param['ens_stats_plots'][field]['ob_plot']['use']:
            plot_obj.plot(ob_coord[:, 0], ob_coord[:, 1], plt_kw=param['ens_stats_plots'][field]['ob_plot']['kw'])

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


def plot_horiz_postage_stamp(ens_obj, param, upp_field='bgd_TCDC_P0_L105_GLC0', klvl=0, 
                             ob_coord=[], save_dir=None, debug=0):
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
        Vertical level to plot. Ignored for 2D fields.
    ob_coord : np.array, optional
        Observation coordinates, (lon, lat)
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
    
    # Set klvl to NaN if 2D field
    mem = ens_obj.mem_names[0]
    if len(ens_obj.subset_ds[mem][upp_field].shape) == 2:
        if debug > 1: print('  2D field. Setting klvl to NaN')
        klvl = np.nan

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
    if param['postage_stamp_plots'][upp_field]['ob_plot']['use']:
        for ax in fig.axes:
            if type(ax) == cartopy.mpl.geoaxes.GeoAxes:
                ax.plot(ob_coord[:, 0], ob_coord[:, 1], transform=ccrs.PlateCarree(),
                        **param['postage_stamp_plots'][upp_field]['ob_plot']['kw'])

    # Save output
    save_fname = f"{save_dir}/postage_stamp_{param['postage_stamp_plots'][upp_field]['save_tag']}_{param['save_tag']}.png"
    plt.savefig(save_fname)

    return fig


def plot_skewt_postage_stamp(ens_obj, param, lat, lon):
    """
    Plot skew-Ts for each ensemble member in a postage stamp plot

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    param : dictionary
        YAML inputs
    lat : float
        Latitude of skew-T (deg N)
    lon : float
        Longitude of skew-T (deg E)

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure containing the Skew-T plot

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
        for prefix, c, in zip(['bgd_', 'ana_'], ['b', 'r']):
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


def plot_cld_obs(ens_obj, cld_ob_df, param, bins=np.arange(0, 2001, 250), nrows=2, ncols=4, figsize=(10, 10), 
                 hofx_kw={}, scatter_kw={}):
    """
    Plot ceilometer obs in horizontal slices for various vertical bins

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
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

    # Run forward operator to add "clear" obs
    cld_hofx = cfo.ceilometer_hofx_driver(cld_ob_df, ens_obj.subset_ds[ens_obj.mem_names[0]], **hofx_kw)

    # Make figure
    fig = plt.figure(figsize=figsize)
    axes = []

    # Plot station IDs in first subplot
    axes.append(fig.add_subplot(nrows, ncols, 1, projection=ccrs.LambertConformal()))
    for i in range(len(cld_hofx.data['HOCB'])):
        axes[-1].text(cld_hofx.data['lon'][i], cld_hofx.data['lat'][i], cld_hofx.data['SID'][i][1:], 
                      size=5, horizontalalignment='center', transform=ccrs.PlateCarree())

    # Plot cloud amounts
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
        axes.append(fig.add_subplot(nrows, ncols, i+2, projection=ccrs.LambertConformal()))
        cax = axes[-1].scatter(obs['lon'], obs['lat'], c=obs['ob_cld_amt'], transform=ccrs.PlateCarree(),
                               **scatter_kw)
        axes[-1].set_title(f"[{bins[i]:.0f}, {bins[i+1]:.0f})", size=14)
    
    # Format subplots
    borders = cfeature.NaturalEarthFeature(category='cultural',
                                           scale='50m',
                                           facecolor='none',
                                           name='admin_1_states_provinces')
    for ax in axes:
        ax.set_extent([param['min_lon'], param['max_lon'], param['min_lat'], param['max_lat']])
        ax.coastlines('50m', edgecolor='gray', linewidth=0.25)
        ax.add_feature(borders, linewidth=0.25, edgecolor='gray')
    
    cbar = plt.colorbar(cax, ax=axes, orientation='vertical')
    cbar.set_label('observed cloud percentage', size=14)

    return fig


def plot_ceil_obs(cld_ob_df, param, figsize=(6, 6), scatter_kw={}):
    """
    Plot observed cloud ceilings

    Parameters
    ----------
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    param : dictionary
        YAML inputs
    figsize : tuple, optional
        Figure size, by default (6, 6)
    scatter_kw : dict, optional
        Keyword arbuments passed to plt.scatter(), by default {}

    Returns
    -------
    fig : plt.figure()
        Plot with desired figure

    """

    # Make figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

    # Plot observed ceilings
    cax = ax.scatter(cld_ob_df['XOB'] - 360, cld_ob_df['YOB'], c=cld_ob_df['CEILING'], transform=ccrs.PlateCarree(),
                     **scatter_kw)
    
    # Add annotations
    borders = cfeature.NaturalEarthFeature(category='cultural',
                                           scale='50m',
                                           facecolor='none',
                                           name='admin_1_states_provinces')
    ax.set_extent([param['min_lon'], param['max_lon'], param['min_lat'], param['max_lat']])
    ax.coastlines('50m', edgecolor='gray', linewidth=0.5)
    ax.add_feature(borders, linewidth=0.5, edgecolor='gray')
    
    cbar = plt.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('cloud ceiling (m)', size=14)

    return fig


def plot_obs_driver(cld_ob_df, ens_obj, ens_z1d, param):
    """
    Run functions to plot cloud observations

    Parameters
    ----------
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    ens_z1d : array
        Average vertical level heights (m)
    param : dictionary
        YAML inputs
    
    Returns
    -------
    None

    """

    # Plot cloud amounts in various vertical bins
    bins = [0] + list(0.5*(ens_z1d[param['plot_stat_config']['klvls']][1:] + 
                           ens_z1d[param['plot_stat_config']['klvls']][:-1]))
    fig = plot_cld_obs(ens_obj, cld_ob_df, param, bins=bins, 
                       nrows=param['plot_stat_config']['nrows'], 
                       ncols=param['plot_stat_config']['ncols'],
                       scatter_kw={'vmin':0, 'vmax':100, 'cmap':'plasma_r', 's':32, 'edgecolors':'k', 'linewidths':0.5})
    plt.savefig(f"{param['out_dir']}/obs_clouds.png", dpi=500)
    plt.close(fig)

    # Plot observed cloud ceilings
    fig = plot_ceil_obs(cld_ob_df, param, scatter_kw=param['obs_plots']['ceil'])
    plt.savefig(f"{param['out_dir']}/obs_ceilings.png", dpi=500)
    plt.close(fig)


def plot_driver(ens_obj, param, save_dir, ob_coord, ens_zlvls, ens_z1d, verbose=0):
    """
    Run code to plot Skew-Ts, ensemble statistics, and postage stamp plots

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    param : dictionary
        YAML inputs
    save_dir : string
        Directory to save figures to
    ob_coord : 2D np.array
        Observation coordinates. Dimensions: (nobs, [z, lon, lat])
    ens_zlvls : 1D np.array
        Integer values corresponding to the model vertical levels
    ens_z1d: 1D np.array
        Average height for each model vertical level
    verbose: integer, optional
        Verbosity level
    
    Returns
    -------
    None

    """

    # Make Skew-T postage stamp plots for the first observation location
    if param['plot_postage_config']['skewt']:
        if verbose > 0: print('Making Skew-T diagram postage stamps...')
        fig = plot_skewt_postage_stamp(ens_obj, param, ob_coord[0, 2], ob_coord[0, 1])
        plt.savefig(f"{save_dir}/postage_stamp_skewt_{param['save_tag']}.pdf")  # Save as a PDF to make it easier to zoom in
        plt.close(fig)

    # Plot ensemble mean and standard deviation
    if verbose > 0: print('Making ensemble statistic plots')
    for field in param['ens_stats_plots']:
        if field not in ens_obj.subset_ds[ens_obj.mem_names[0]]:
            if verbose > 0: print(f'  field {field} is missing. Skipping.')
            continue
        if ('bgd' in field) and (param['plot_stat_config']['plot_bgd_once']):
            if verbose > 0: print(f'  skipping background field {field}')
            continue
        if verbose > 0: print(f'  plotting {field}...')
        fig = plot_horiz_slices(ens_obj.subset_ds[ens_obj.mem_names[0]], 
                                field,
                                ens_obj,
                                param,
                                ob_coord=ob_coord[:, 1:],
                                save_dir=save_dir)
        plt.close(fig)

    # Make postage stamp plots
    if verbose > 0: print('Making postage stamp plots')
    klvl = np.argmin(np.abs(ens_zlvls - np.mean(ob_coord[:, 0])))
    if verbose > 0: print(f"postage stamp klvl = {klvl} ({ens_z1d[klvl]} m AGL)")
    for field in param['postage_stamp_plots'].keys():
        if field not in ens_obj.subset_ds[ens_obj.mem_names[0]]:
            print(f'  field {field} is missing. Skipping.')
            continue
        if verbose > 0: print(f'  plotting {field}...')
        fig = plot_horiz_postage_stamp(ens_obj, param, upp_field=field, 
                                       klvl=klvl,
                                       ob_coord=ob_coord[:, 1:],
                                       save_dir=save_dir,
                                       debug=0)
        plt.close(fig)


"""
End cloud_DA_enkf_viz.py
"""