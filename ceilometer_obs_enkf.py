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

import main.cloud_DA_forward_operator as cfo
import main.cloud_DA_enkf_postprocess as ens_post
from main import ens_io
from main import obs_io
from pyDA_utils import enkf
import pyDA_utils.ensemble_utils as eu
from pyDA_utils import bufr
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

    fnames = [param['ens']['path'].format(num=n) for n in range(1, param['nmem'] + 1)]
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
                                   verbose=param['obs']['verbose'])
    
    # Remove missing cloud obs
    obs_df = cfo.remove_missing_cld_ob(bufr_df)

    # Only retain obs from desired stations
    if not param['obs']['entire_file']:
        SIDs = list(param['DA']['ob_sel'].keys())
        cond = np.zeros(len(obs_df))
        for s in SIDs:
            cond = cond + (obs_df['SID'] == s)
        if np.sum(cond) == 0:
            raise ValueError("read_obs(): entire_file = False, but no valid sites selected for DA")
        obs_df = obs_df.loc[cond > 0, :]
    
    return obs_df


def run_cld_forward_operator(ens_obj, cld_ob_df, hofx_kw={}, cld_field='cld_frac', verbose=False):
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
    cld_field : string, optional
        Name of cloud fraction field
    verbose : boolean, optional
        Option to print extra output
    
    Returns
    -------
    cld_hofx : dictionary of cfo.sfc_cld_forward_operator objects
        Ceilometer forward operator output for each ensemble member

    """
    
    cld_hofx = {}

    # Run forward operator
    for n in range(ens_obj.meta['Nens']):
        if verbose: print(f'Running forward operator on ensemble member {n+1}')
        model_dict = ens_obj.var_dict(n)
        cld_hofx[n] = cfo.ceilometer_hofx_driver(cld_ob_df, model_dict, **hofx_kw)
    
    return cld_hofx


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


def run_enkf(ens_obj, ob_df, param, verbose=0):
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
    verbose : int, optional
        Verbosity level

    Returns
    -------
    ens_obj : ens_io.ens_data object
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
    ens_obj, ob_coord_all = run_enkf(ens_obj, cld_ob_df, param, verbose=param['DA']['verbose'])
    ens_obj = ens_post.post_enkf(ens_obj, param, DA=param['perform_da'])
    
    print(f'\ntotal elapsed time = {(dt.datetime.now() - start).total_seconds()} s')


"""
End ceilometer_obs_enkf.py
"""
