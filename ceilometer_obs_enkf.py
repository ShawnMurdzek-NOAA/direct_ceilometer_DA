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

    if 'n_zlvl' not in list(param['ens'].keys()):
        param['ens']['n_zlvl'] = None

    fnames = [param['ens']['in_path'].format(num=n) for n in range(1, param['ens']['nmem'] + 1)]
    ens_obj = ens_io.read_ens(fnames,
                              state_fields=param['DA']['state_vars'],
                              other_fields={},
                              verbose=param['ens']['verbose'],
                              fix_fname=param['ens']['fix_file'],
                              ftype=param['ens']['type'],
                              k_end=param['ens']['n_zlvl'])

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
    cld_hofx : dictionary
        Series of fields corresponding to H(x) output, including...
            hofx : H(x) value. Dimensions: (Nobs, Nens)
            ob : Observed value. Dimensions: (Nobs)
            loc : Observation location. Dimensions: (Nobs, 3)
            SID : Station IDs. Dimensions: (Nobs)
            HOCB : Observation heights in m. Dimensions: (Nobs)

    """
    
    cld_hofx_ls = []

    # Run forward operator
    if Nens == 0: Nens = ens_obj.meta['Nens']
    for n in range(Nens):
        if verbose > 0: print(f'Running forward operator on ensemble member {n+1}')
        model_dict = ens_obj.var_dict(n)
        cld_hofx_ls.append(cfo.ceilometer_hofx_driver(cld_ob_df, model_dict, **hofx_kw))
   
    # Reformat output
    Nobs = 0
    Nsid = len(cld_hofx_ls[0].data['SID'])
    for i in range(Nsid):
        Nobs = Nobs + len(cld_hofx_ls[0].data['HOCB'][i])

    cld_hofx = {'hofx': np.zeros([Nobs, Nens]),
                'ob': np.zeros([Nobs]),
                'loc': np.zeros([Nobs, 3]),
                'SID': [],
                'HOCB': np.zeros([Nobs])}

    n = 0
    for i in range(Nsid):
        nz = len(cld_hofx_ls[0].data['HOCB'][i])
        cld_hofx['ob'][n:(n+nz)] = cld_hofx_ls[0].data['ob_cld_amt'][i]
        cld_hofx['loc'][n:(n+nz), 1] = cld_hofx_ls[0].data['lat'][i]
        cld_hofx['loc'][n:(n+nz), 2] = cld_hofx_ls[0].data['lon'][i]
        cld_hofx['SID'] = cld_hofx['SID'] + [cld_hofx_ls[0].data['SID'][i]] * nz
        cld_hofx['HOCB'][n:(n+nz)] = cld_hofx_ls[0].data['HOCB'][i]
        for j in range(Nens):
            cld_hofx['loc'][n:(n+nz), 0] = cld_hofx['loc'][n:(n+nz), 0] + cld_hofx_ls[j].data['ob_hgt_model'][i]
            cld_hofx['hofx'][n:(n+nz), j] = cld_hofx_ls[j].data['hofx'][i]
        n = n + nz

    cld_hofx['SID'] = np.array(cld_hofx['SID'])
    cld_hofx['loc'][:, 0] = cld_hofx['loc'][:, 0] / Nens

    return cld_hofx


def compute_localization_model(ens_obj, param, z, lat, lon):
    """
    Compute localization array for the model gridpoints

    Parameters
    ----------
    ens_obj : ens_io.ens_data object
        Ensemble output
    param : dictionary
        YAML inputs
    z : float
        Observation height
    lat : float
        Observation latitude (deg N)
    lon : float
        Observation longitude (deg E)
    
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


def compute_localization_hofx(hofx_pts, param, z, lat, lon):
    """
    Compute localization array for H(x)

    Parameters
    ----------
    hofx_pts : array
        Coordinates of H(x) values in (z, lat, lon). Dimensions: (npts, 3)
    param : dictionary
        YAML inputs
    z : float
        Observation height
    lat : float
        Observation latitude (deg N)
    lon : float
        Observation longitude (deg E)

    Returns
    -------
    C : np.ndarray
        Localization array

    """

    # Use Gaspari and Cohn (1999) 5th-order localization fct
    local_fct = local.localization_fct(local.gaspari_cohn_5ord)

    # Extract information needed to compute localization
    lh = param['DA']['localization']['lh']
    lv = param['DA']['localization']['lv']

    ob_pt = np.array([z, lat, lon])
    C = local_fct.compute_localization(hofx_pts, ob_pt, lv, lh)

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

    # Apply cloud DA forward operator
    cld_hofx = run_cld_forward_operator(ens_obj, ob_df, hofx_kw=param['DA']['hofx_kw'], 
                                        verbose=param['DA']['verbose'], Nens=0)
    if param['DA']['verbose'] > 0: print(f"Time to complete forward operator for all members and obs = {(dt.datetime.now() - start_enkf).total_seconds()} s")

    # Save some diagnostics
    Nobs, Nens = cld_hofx['hofx'].shape
    diag = {}
    diag['hgt'] = cld_hofx['HOCB']
    diag['lat'] = cld_hofx['loc'][:, 1]
    diag['lon'] = cld_hofx['loc'][:, 2]
    diag['ob'] = cld_hofx['ob']
    diag['use'] = np.ones(Nobs)
    for k in range(Nens):
        diag[f"omb{k+1}"] = cld_hofx['ob'] - cld_hofx['hofx'][:, k]

    # Loop over each observation
    for i, s in enumerate(cld_hofx['SID']):

        start_loop = dt.datetime.now()

        # Option to only assimilate certain obs from a particular SID
        if (not param['obs']['entire_file']) and (len(param['obs']['ob_sel'][s]) > 0):
            all_sid_idx = np.where(cld_hofx['SID'] == s)[0]
            this_sid_idx = np.where(all_sid_idx == i)[0][0]
            if this_sid_idx not in param['obs']['ob_sel'][s]:
                diag['use'][i] = 0
                continue

        if param['DA']['verbose'] > 1: 
            print(f"  -----------------------")
            print(f"  Station = {s}, Ob = {i} (of {Nobs})")

        # Extract cloud amount, H(x), and location
        start_extract = dt.datetime.now()
        hofx = cld_hofx['hofx'][i, :]
        cld_ob_coord = cld_hofx['loc'][i, :]
        cld_amt = cld_hofx['ob'][i]
        if param['DA']['verbose'] > 2: print(f"  Time to extract cld amt, H(x), and location = {(dt.datetime.now() - start_extract).total_seconds()} s")
        if param['DA']['verbose'] > 2: print("  ob loc =", cld_ob_coord)
        if param['DA']['verbose'] > 2: print("  ob =", cld_amt)
        if param['DA']['verbose'] > 2: print("  H(x) =", hofx)

        # Skip remaining steps if not performing DA or if all O-B values are 0
        omb = cld_amt - hofx
        if (not param['DA']['perform_da']) or (np.isclose(np.sum(np.abs(omb)), 0) and param['DA']['skip_zero_omb']):
            if param['DA']['verbose'] > 1: print("  Skipping DA step")
            diag['use'][i] = 0
            continue
            
        # Compute localization
        if param['DA']['localization']['use']:
            start_local = dt.datetime.now()
            if param['DA']['verbose'] > 2: print(f"  computing localization with lh = {param['DA']['localization']['lh']}, lv = {param['DA']['localization']['lv']}")
            C_local = compute_localization_model(ens_obj, param, cld_ob_coord[0], cld_ob_coord[1], cld_ob_coord[2])
            if param['DA']['update_hofx_with_enkf']:
                C_hofx = compute_localization_hofx(cld_hofx['loc'], param, cld_ob_coord[0], cld_ob_coord[1], cld_ob_coord[2])
            if param['DA']['verbose'] > 1: print(f"  Time to complete localization = {(dt.datetime.now() - start_local).total_seconds()} s")
        else:
            C_local = None
            C_hofx = None

        # Run EnKF
        start_ensrf = dt.datetime.now()
        enkf_obj = enkf.enkf_1ob(ens_obj.state, cld_amt, hofx, param['DA']['ob_var'], localize=C_local)
        enkf_obj.EnSRF()
        if param['DA']['update_hofx_with_enkf']:
            enkf_obj_hofx = enkf.enkf_1ob(cld_hofx['hofx'], cld_amt, hofx, param['DA']['ob_var'], localize=C_hofx)
            enkf_obj_hofx.EnSRF()
        if param['DA']['verbose'] > 1: print(f"  Time to complete EnSRF = {(dt.datetime.now() - start_ensrf).total_seconds()} s")

        # Update ens_obj with the new analysis
        ens_obj.state = enkf_obj.x_a
        if param['DA']['update_hofx_with_enkf']:
            cld_hofx['hofx'] = enkf_obj_hofx.x_a

        if param['DA']['verbose'] > 0: print(f"  Time to assimilate {i} = {(dt.datetime.now() - start_loop).total_seconds()} s")

    # Compute O-A and save diagnostics to DataFrame
    cld_hofxa = run_cld_forward_operator(ens_obj, ob_df, hofx_kw=param['DA']['hofx_kw'], 
                                         verbose=param['DA']['verbose'], Nens=0)
    for k in range(ens_obj.meta['Nens']):
        diag[f"oma{k+1}"] = cld_hofxa['ob'] - cld_hofxa['hofx'][:, k] 
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
