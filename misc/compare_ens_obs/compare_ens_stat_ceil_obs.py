"""
Compare Various Ensemble Statistics to Ceilometer Observations

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime as dt

import pyDA_utils.ensemble_utils as eu


#---------------------------------------------------------------------------------------------------
# Main Program
#---------------------------------------------------------------------------------------------------

def read_ensemble_output_1time(param, time, verbose=1):
    """
    Reads UPP pressure-lebel ensemble output for a single time

    Parameters
    ----------
    param : dictionary
        YAML inputs
    time : dt.datetime object
        Init time for ensemble (1-hr forecasts are extracted)
    verbose : int, optional
        Verbosity level, by default 1

    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object

    """

    start = dt.datetime.now()

    # UPP pressure-level output names
    str_format = param['upp_format']
    fnames = {}
    for i in range(1, param['nmem']+1):
        fnames['mem{num:04d}'.format(num=i)] = str_format.format(day=time.strftime("%Y%m%d"), 
                                                                 hr=time.strftime("%H"), 
                                                                 num=i)
    
    # Verifying observation file name
    obs_format = param['bufr_format']
    obs_fname = obs_format.format(date=time.strftime("%Y%m%d%H%M"))

    # Read ensemble output
    start = dt.datetime.now()
    ens_obj = eu.ensemble(fnames,  
                          bufr_csv_fname=obs_fname, 
                          zind=list(range(100000, 9000, -2500)), 
                          zfield='lv_ISBL0')
    if verbose > 0:
        print('I/O elapsed time = {t:.2f} s'.format(t=(dt.datetime.now() - start).total_seconds()))
        print('Shape of subset =', ens_obj.subset_ds['mem0001']['CEIL_P0_L2_GLC0'].shape)

    return ens_obj


def interp_ens_to_obs(ens_obj, param):
    """
    Interpolate ensemble output to ceiling observation locations

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble
        Ensemble object
    param : dictionary
        YAML inputs
    
    Returns
    -------
    ceil_obs : pd.DataFrame
        Ceiling observations
    ceil_ens : pd.DataFrame
        Ceilings from each ensemble member

    """

    # Subset obs
    ceil_obs = ens_obj._subset_bufr(['ADPSFC'], 
                                    nonan_field="CEILING", 
                                    DHR=0)

    # Interpolate ensemble output to obs locations
    ceil_ens = ens_obj.interp_model_2d(param['ceil_field'], 
                                       ceil_obs['YOB'].values, 
                                       ceil_obs['XOB'].values - 360, 
                                       zind=np.nan, 
                                       method='nearest', 
                                       verbose=True)

    return ceil_obs, ceil_ens


def config_ax(ax, param):
    """
    Add coastlines and state borders to an axis and set lat and lon limits

    Parameters
    ----------
    ax : matplotlib.axes
        Axes to modify
    param : dictionary
        YAML inputs

    Returns
    -------
    ax : matplotlib.axes
        Modified axes

    """

    # Add coastlines and state borders
    line_kw={'ls':'-', 'lw':1}
    ax.coastlines('50m', **line_kw)
    borders = cfeature.NaturalEarthFeature(category='cultural',
                                                   scale='50m',
                                                   facecolor='none',
                                                   name='admin_1_states_provinces')
    ax.add_feature(borders, **line_kw)

    # Set lat and lon limits
    ax.set_extent([param['xlim'][0], param['xlim'][1], param['ylim'][0], param['ylim'][1]])

    return ax


if __name__ == '__main__':

    start = dt.datetime.now()
    print('Starting program')

    # Read in input YAML file
    with open(sys.argv[1], 'r') as fptr:
        param = yaml.safe_load(fptr)

    # Create list of datetime objects to loop over
    dt_list = [dt.datetime.strptime(str(d), '%Y%m%d%H') for d in param['dates']]
    
    # Loop over each init time
    for d in dt_list:
        print()
        print(f"time = {d.strftime('%Y%m%d%H')}")
        ens_obj = read_ensemble_output_1time(param, d, verbose=0)
        ceil_obs, ceil_ens = interp_ens_to_obs(ens_obj, param)

        # Compute ensemble stats
        mem_names = list(ens_obj.subset_ds.keys())
        ceil_ens_75 = np.percentile(ceil_ens.loc[:, mem_names].values, 75, axis=1)
        ceil_ens_25 = np.percentile(ceil_ens.loc[:, mem_names].values, 25, axis=1)

        # Plot results
        fig = plt.figure(figsize=(12, 12))

        ax = fig.add_subplot(2, 1, 1, projection=ccrs.LambertConformal())
        idx = np.where(np.logical_and(ceil_ens_75 < 3000, ceil_obs['CEILING'] > 19000))[0]
        ax.plot(ceil_obs['XOB'][idx].values - 360., ceil_obs['YOB'][idx].values, 'b.', transform=ccrs.PlateCarree())
        ax.set_title('Ensemble 75th Percentile Has Spurious Ceilings', size=14)
        ax = config_ax(ax, param)

        ax = fig.add_subplot(2, 1, 2, projection=ccrs.LambertConformal())
        idx = np.where(np.logical_and(ceil_ens_25 > 19000, ceil_obs['CEILING'] < 3000))[0]
        ax.plot(ceil_obs['XOB'][idx].values - 360., ceil_obs['YOB'][idx].values, 'r.', transform=ccrs.PlateCarree())
        ax.set_title('Ensemble 25th Percentile is Missing Ceilings', size=14)
        ax = config_ax(ax, param)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.92)
        plt.suptitle(f"init time = {d.strftime('%Y%m%d%H')}", size=16)
        plt.savefig(f"{d.strftime('%Y%m%d%H')}_{param['out_tag']}.png")

    # Elapsed time
    print('Program Complete!')
    print('Elapsed time = {t:.2f} s'.format(t=(dt.datetime.now() - start).total_seconds()))


"""
End compare_ens_stat_ceil_obs.py
"""
