"""
Plot Cloud Observations from BUFR CSV File

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import datetime as dt
import sys
import argparse
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import yaml

from pyDA_utils import bufr


#---------------------------------------------------------------------------------------------------
# Main Program
#---------------------------------------------------------------------------------------------------

def parse_in_args(argv):
    """
    Parse input arguments

    Parameters
    ----------
    argv : list
        Command-line arguments from sys.argv[1:]
    
    Returns
    -------
    Parsed input arguments

    """

    parser = argparse.ArgumentParser(description="Script that plots cloud observations from a BUFR \
                                                  CSV file. Two plots are created: (1) Cloud \
                                                  ceilings and (2) Cloud amounts in various \
                                                  vertical bins.")
    
    # Positional arguments
    parser.add_argument('bufr_file', 
                        help='CSV file containing BUFR observations',
                        type=str)
    
    parser.add_argument('param_file', 
                        help='YAML file containing plot specifications',
                        type=str)
    
    return parser.parse_args(argv)


def read_param(fname):
    """
    Read input YAML file.

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    param : dictionary
        Input parameters

    """

    with open(fname, 'r') as fptr:
        param = yaml.safe_load(fptr)
    
    return param


def read_bufr(fname, param):
    """
    Read and preprocess BUFR CSV obs

    Parameters
    ----------
    fname : string
        BUFR CSV file name
    param : argparse.Namespace
        Input parameters
    
    Returns
    -------
    ob_df : pd.DataFrame
        Observations to plot
    time : dt.datetime
        BUFR CSV file time

    """

    bufr_csv = bufr.bufrCSV(fname)

    # Only retain desired stations
    if not param['plot_all']:
        bufr_csv.select_SIDs(param['plot_sid'])

    # Determine time for BUFR CSV file
    time = dt.datetime.strptime(str(bufr_csv.df['cycletime'].values[0]), '%Y%m%d%H')
    
    return bufr_csv.df, time


def add_ax_annot(ax, param):
    """
    Add various annotations to a matplotlib axes

    Parameters
    ----------
    ax : matplotlib.axes
        Axes
    param : dictionary
        Plotting parameters
    
    Returns
    -------
    ax : matplotlib.axes
        Axes with annotations

    """

    borders = cfeature.NaturalEarthFeature(category='cultural',
                                           scale='50m',
                                           facecolor='none',
                                           name='admin_1_states_provinces')
    ax.set_extent([param['min_lon'], param['max_lon'], param['min_lat'], param['max_lat']])
    ax.coastlines('50m', linewidth=0.25, edgecolor='gray')
    ax.add_feature(borders, linewidth=0.25, edgecolor='gray')

    return ax


def plot_ceil_obs(cld_ob_df, param):
    """
    Plot observed cloud ceilings

    Parameters
    ----------
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    param : dictionary
        YAML inputs

    Returns
    -------
    fig : plt.figure()
        Plot with desired figure

    """

    # Make figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

    # Plot observed ceilings
    cax = ax.scatter(cld_ob_df['XOB'] - 360, 
                     cld_ob_df['YOB'], 
                     c=cld_ob_df['CEILING'], 
                     transform=ccrs.PlateCarree(), 
                     vmin=param['ceil']['vmin'], 
                     vmax=param['ceil']['vmax'],
                     cmap=param['ceil']['cmap'])
    
    # Add annotations
    ax = add_ax_annot(ax, param)
    cbar = plt.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('cloud ceiling (m)', size=14)

    return fig


def plot_cld_amt_obs(cld_ob_df, param):
    """
    Plot ceilometer obs in horizontal slices for various vertical bins

    Parameters
    ----------
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    param : dictionary
        YAML inputs

    Returns
    -------
    fig : plt.figure()
        Plot with desired figure

    """

    # Make figure
    fig = plt.figure(figsize=param['cld_amt']['figsize'])
    axes = []

    # Plot station IDs in first subplot
    axes.append(fig.add_subplot(param['cld_amt']['nrows'], 
                                param['cld_amt']['ncols'], 
                                1, 
                                projection=ccrs.LambertConformal()))
    obs = {'sid': np.unique(cld_ob_df['SID'].values)}
    obs['lat'] = np.zeros(len(obs['sid'])) 
    obs['lon'] = np.zeros(len(obs['sid'])) 
    for i, sid in enumerate(obs['sid']):
        obs['lon'][i] = cld_ob_df.loc[cld_ob_df['SID'] == sid, 'XOB'].values[0] - 360
        obs['lat'][i] = cld_ob_df.loc[cld_ob_df['SID'] == sid, 'YOB'].values[0]
        axes[-1].text(obs['lon'][i], 
                      obs['lat'][i], 
                      sid, 
                      size=6, 
                      horizontalalignment='center', 
                      transform=ccrs.PlateCarree())

    # Plot cloud amounts
    bins = param['cld_amt']['bins']
    for i in range(len(bins) - 1):

        axes.append(fig.add_subplot(param['cld_amt']['nrows'], 
                                    param['cld_amt']['ncols'], 
                                    i+2, 
                                    projection=ccrs.LambertConformal()))
        
        # Create a DataFrame with cloud obs in this height bin
        subset_df = cld_ob_df.loc[np.logical_and(cld_ob_df['HOCB'].values >= bins[i],
                                                 cld_ob_df['HOCB'].values < bins[i+1]), ['CLAM', 'HOCB', 'SID']]

        # Plot CLAM values for this height bin
        for j, sid in enumerate(obs['sid']):
            clam = 0
            if sid in subset_df['SID'].values:
                clam = subset_df.loc[subset_df['SID'] == sid, 'CLAM'].values[0]
            axes[-1].text(obs['lon'][j], 
                          obs['lat'][j],
                          int(clam), 
                          size=8, 
                          horizontalalignment='center', 
                          transform=ccrs.PlateCarree(),
                          color='r')

        axes[-1].set_title(f"CLAM: [{bins[i]:.0f}, {bins[i+1]:.0f}) m", size=14)
    
    # Format subplots
    for ax in axes:
        ax = add_ax_annot(ax, param)

    return fig


if __name__ == '__main__':

    start = dt.datetime.now()
    print('Starting plot_bufr_cloud_obs.py')
    print(f"Time = {start.strftime('%Y%m%d %H:%M:%S')}")

    clargs = parse_in_args(sys.argv[1:])
    param = read_param(clargs.param_file)

    print('Reading in BUFR data...')
    cld_ob_df, time = read_bufr(clargs.bufr_file, param)

    print('Creating cloud amount plot...')
    fig = plot_cld_amt_obs(cld_ob_df, param)
    plt.savefig(f"cld_amt_{time.strftime('%Y%m%d%H')}_{param['plot_tag']}.png")
    plt.close()

    print('Creating ceiling plot...')
    fig = plot_ceil_obs(cld_ob_df, param)
    plt.savefig(f"ceil_{time.strftime('%Y%m%d%H')}_{param['plot_tag']}.png")
    plt.close()

    print('Program Finished!')
    print(f"Elapsed time = {(dt.datetime.now() - start).total_seconds()} s")


"""
End plot_bufr_cloud_obs.py
"""