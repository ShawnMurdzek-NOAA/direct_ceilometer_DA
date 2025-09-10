"""
Ceilometer Observation DA Tests Using an EnKF

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import numpy as np
import datetime as dt
import copy
import yaml
import pandas as pd

import main.cloud_DA_forward_operator as cfo
from main import ens_io
from main import obs_io
from pyDA_utils import enkf
import pyDA_utils.localization as local


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def read_param(fname):
    """
    Read input YAML file and reformat plotting options in input parameters.

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    param : dictionary
        Input parameters

    """

    # Read input parameters
    with open(fname, 'r') as fptr:
        param = yaml.safe_load(fptr)
    
    return param


def read_ensemble(param):
    """
    Read ensemble data

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    ens_obj : ens_io.ens_data object
        Ensemble data

    """

    if (param['ens']['type'] == 'upp'):
        print('\nWarning: UPP output has not been extensively tested yet\n')

    fnames = [param['ens']['in_path'].format(num=n) for n in range(1, param['ens']['nmem'] + 1)]
    ens_obj = ens_io.read_ens(fnames,
                              state_fields=param['DA']['state_vars'],
                              other_fields={},
                              verbose=param['ens']['verbose'],
                              fix_fname=param['ens']['fix_file'],
                              ftype=param['ens']['type'])

    return ens_obj


def read_obs(param):
    """
    Read and preprocess observational data

    Parameters
    ----------
    param : dictionary
        Input parameters
    
    Returns
    -------
    obs_df : pd.DataFrame
        Observations

    """

    # Read obs
    bufr_df = obs_io.read_bufr_obs(param['obs']['fname'],
                                   subset=['ADPSFC', 'MSONET'],
                                   domain=param['obs']['domain'],
                                   lim_DHR=param['obs']['lim_DHR'],
                                   verbose=param['obs']['verbose'])
    
    # Remove missing cloud obs
    obs_df = cfo.remove_missing_cld_ob(bufr_df)

    # Only retain obs from desired stations
    if not param['obs']['entire_file']:
        SIDs = list(param['obs']['ob_sel'].keys())
        cond = np.zeros(len(obs_df))
        for s in SIDs:
            cond = cond + (obs_df['SID'] == s)
        if np.sum(cond) == 0:
            raise ValueError("read_obs(): entire_file = False, but no valid sites selected for DA")
        obs_df = obs_df.loc[cond > 0, :]
    
    return obs_df


def run_cld_forward_operator(ens_obj, cld_ob_df, hofx_kw={}, verbose=0, Nens=0):
    """
    Run the cloud DA forward operator for all observations in the subset domain

    Parameters
    ----------
    ens_obj : ens_io.ens_data object
        Ensemble output
    cld_ob_df : pd.DataFrame
        Ceilometer observations used in the forward operator
    hofx_kw : dictionary, optional
        Keyword arguments passed to cfo.ceilometer_hofx_driver()
    verbose : integer, optional
        Option to print extra output
    Nens : integer, optional
        Option to only run forward operator on the first Nens ensemble members. Set to 0 to run on
        all ensemble members
    
    Returns
    -------
    cld_hofx : dictionary of cfo.sfc_cld_forward_operator objects
        Ceilometer forward operator output for each ensemble member

    """
    
    cld_hofx = []

    # Run forward operator
    if Nens == 0: Nens = ens_obj.meta['Nens']
    for n in range(Nens):
        if verbose > 0: print(f'Running forward operator on ensemble member {n+1}')
        model_dict = ens_obj.var_dict(n)
        cld_hofx.append(cfo.ceilometer_hofx_driver(cld_ob_df, model_dict, **hofx_kw))
    
    return cld_hofx


def compute_localization_array(ens_obj, param, z, lon, lat):
    """
    Compute localization array for EnKF DA

    Parameters
    ----------
    ens_obj : ens_io.ens_data object
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
    Nz = ens_obj.meta['Nz']
    N2d = ens_obj.meta['N2d']
    Nvar = ens_obj.meta['Nvars']
    lh = param['DA']['localization']['lh']
    lv = param['DA']['localization']['lv']

    # Compute localization in horizontal and vertical dimensions separately, then combine
    # Note that localization in the vertical uses the model vertical level (not height)
    model_latlon_pts = np.array([ens_obj.loc['lat'], ens_obj.loc['lon']]).T
    model_z_pts = np.arange(Nz)
    Ch = local_fct.compute_partial_localization(model_latlon_pts, [lat, lon], lh)
    Cv = local_fct.compute_partial_localization(model_z_pts, z, lv)
    C_3d_1var = np.repeat(Cv[np.newaxis, :], N2d, axis=0) * np.repeat(Ch[:, np.newaxis], Nz, axis=1)
    C = np.array(list(np.ravel(C_3d_1var)) * Nvar)

    # Old approach - Much slower, especially for larger domains
    # Current approach and old approach do not perfectly agree, but for the test case, max diffs
    # tend to be O(1e-5), which I think are acceptable
    #model_pts = np.array([list(np.ravel(np.repeat(np.arange(Nz)[np.newaxis, :], N2d, axis=0))) * Nvar, 
    #                      list(np.ravel(np.repeat(ens_obj.loc['lat'][:, np.newaxis], Nz, axis=1))) * Nvar, 
    #                      list(np.ravel(np.repeat(ens_obj.loc['lon'][:, np.newaxis], Nz, axis=1))) * Nvar]).T
    #ob_pt = np.array([z, lat, lon])
    #C_old = local_fct.compute_localization(model_pts, ob_pt, lv, lh)
    #print(np.amax(np.abs(C - C_old)))

    return C


def run_enkf(ens_obj, ob_df, param):
    """
    Run EnKF for an arbitrary number of observations

    Parameters
    ----------
    ens_obj : ens_io.ens_data object
        Ensemble output
    ob_df : pd.DataFrame
        Ceilometer observations
    param : dictionary
        Input YAML parameters

    Returns
    -------
    ens_obj : ens_io.ens_data object
        Ensemble output
    diag_df : pd.DataFrame
        DA diagnostic data (e.g., ob, H(x), O-B, etc.)
        
    """

    start_enkf = dt.datetime.now()

    # Initialize dictionary for diagnostic output
    diag = {}
    for f in ['hgt', 'lon', 'lat', 'ob']:
        diag[f] = []
    for f in ['omb', 'oma']:
        for k in range(ens_obj.meta['Nens']):
            diag[f"{f}{k+1}"] = []

    # Apply cloud DA forward operator
    cld_hofx = run_cld_forward_operator(ens_obj, ob_df, hofx_kw=param['DA']['hofx_kw'], 
                                        verbose=param['DA']['verbose'], Nens=0)
    if param['DA']['verbose'] > 0: print(f"Time to complete forward operator for all members and obs = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    # Loop over each observation
    for i, s in enumerate(cld_hofx[0].data['SID']):
        if (param['obs']['entire_file']) or (len(param['obs']['ob_sel'][s]) == 0):
            ob_idx = list(range(len(cld_hofx[0].data['HOCB'][i])))
        else:
            ob_idx = param['obs']['ob_sel'][s]
        for j in ob_idx:
            start_loop = dt.datetime.now()
            if param['DA']['verbose'] > 1: print(f"  Looping over ob {s} {j}")

            # Extract cloud amount, H(x), and location
            hofx = np.zeros(ens_obj.meta['Nens'])
            cld_ob_coord = [0, cld_hofx[0].data['lon'][i], cld_hofx[0].data['lat'][i]]
            for k in range(ens_obj.meta['Nens']):
                hofx[k] = cld_hofx[k].data['hofx'][i][j]
                cld_ob_coord[0] = cld_ob_coord[0] + cld_hofx[k].data['ob_hgt_model'][i][j]
            cld_ob_coord[0] = cld_ob_coord[0] / ens_obj.meta['Nens']
            cld_amt = cld_hofx[0].data['ob_cld_amt'][i][j]

            if param['DA']['verbose'] > 2: print("  H(x) =", hofx)

            # Save diagnostic output
            diag['hgt'].append(cld_hofx[0].data['HOCB'][i][j])    # Height in m rather than height in model vertical levels
            diag['lon'].append(cld_ob_coord[1])
            diag['lat'].append(cld_ob_coord[2])
            diag['ob'].append(cld_amt)
            for k in range(ens_obj.meta['Nens']):
                diag[f"omb{k+1}"].append(cld_amt - hofx[k])

            # Skip remaining steps if not performing DA
            if not param['DA']['perform_da']:
                continue
            
            # Compute localization
            if param['DA']['localization']['use']:
                start_local = dt.datetime.now()
                if param['DA']['verbose'] > 2: print(f"  computing localization with lh = {param['DA']['localization']['lh']}, lv = {param['DA']['localization']['lv']}")
                C_local = compute_localization_array(ens_obj, param, cld_ob_coord[0], cld_ob_coord[1], cld_ob_coord[2])
                if param['DA']['verbose'] > 0: print(f"  Time to complete localization = {(dt.datetime.now() - start_local).total_seconds()} s")
            else:
                C_local = None

            # Run EnKF
            enkf_obj = enkf.enkf_1ob(ens_obj.state, cld_amt, hofx, param['DA']['ob_var'], localize=C_local)
            enkf_obj.EnSRF()

            # Update ens_obj with the new analysis
            ens_obj.state = enkf_obj.x_a

            if param['DA']['verbose'] > 0: print(f"  Time to assimilate {s} {j} = {(dt.datetime.now() - start_loop).total_seconds()} s")

    # Compute O-A and save diagnostics to DataFrame
    cld_hofxa = run_cld_forward_operator(ens_obj, ob_df, hofx_kw=param['DA']['hofx_kw'], 
                                         verbose=param['DA']['verbose'], Nens=0)
    for k in range(ens_obj.meta['Nens']):
        cld_hofxa[k].compute_OmB()
    for i, s in enumerate(cld_hofx[0].data['SID']):
        if (param['obs']['entire_file']) or (len(param['obs']['ob_sel'][s]) == 0):
            ob_idx = list(range(len(cld_hofx[0].data['HOCB'][i])))
        else:
            ob_idx = param['obs']['ob_sel'][s]
        for j in ob_idx:
            for k in range(ens_obj.meta['Nens']):
                diag[f"oma{k+1}"].append(cld_hofxa[k].data['OmB'][i][j])
    diag_df = pd.DataFrame(diag)

    if param['DA']['verbose'] > 0: print(f"run_enkf total time = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    return ens_obj, diag_df


def save_ens(ens_obj, param):
    """
    Save output form EnKF to a netCDF file

    Parameters
    ----------
    ens_obj : ens_io.ens_data object
        Ensemble data after performing EnKF
    param : dictionary
        Input parameters
    
    Returns
    -------
    None

    """

    in_fnames = [param['ens']['in_path'].format(num=n) for n in range(1, param['ens']['nmem'] + 1)]
    out_fnames = [param['ens']['out_path'].format(num=n) for n in range(1, param['ens']['nmem'] + 1)]

    ens_obj.write_mpas_out_for_DA(in_fnames, out_fnames)


if __name__ == '__main__':

    start = dt.datetime.now()
    print('\n-----------------------------------------------')
    print(f"Starting Cloud DA Program")

    # Read input YAML file
    param = read_param(sys.argv[1])

    # Read ensemble data and observations
    print('\nReading ensemble data')
    ens_obj = read_ensemble(param)
    print('Reading observations')
    cld_ob_df = read_obs(param)

    # Run EnKF
    print('\nRunning EnKF')
    ens_obj, diag_df = run_enkf(ens_obj, cld_ob_df, param)

    # Save output
    print('\nWriting output to netCDF file')
    save_ens(ens_obj, param)
    print('Writing DA diagnostic output')
    diag_df.to_csv(param['DA']['diag_file'])
    
    print(f'\ntotal elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End ceilometer_obs_enkf.py
"""
