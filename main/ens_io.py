"""
Functions for Cloud DA Ensemble I/O

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import numpy as np

import pyDA_utils.ensemble_utils as eu


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class ens_data():
    """
    Class to handle ensemble output for cloud DA

    Parameters
    ----------
    state : np.array
        State matrix used for DA. Dimensions: (Nx, Nens)
    varnames : np.array
        Names of the forecast variables in state. Dimensions: (Nvars)
    loc : dictionary
        Location of forecast variables. Must include:
            lat : Latitude in deg N. Dimensions: (N2d)
            lon : Longitude in deg E. Dimensions: (N2d)
            hgt : Height AGL (m). Dimensions: (N2d, Nz)
        Note: N2d * Nz * Nvars = Nx
    other : dictionary, optional
        Other fields that are not part of the state matrix. Might be needed for H(x). 
        Key: field name. Value has dimensions (N2d * Nz, Nens)
    meta : dictionary, optional
        Metadata

    """

    def __init__(self, state, varnames, loc, other={}, meta={}):

        self.state = state
        self.varnames = varnames
        self.loc = loc
        self.other = other
        self.meta = meta

        # Determine number of forecast variables and ensemble members
        self.meta['Nx'], self.meta['Nens'] = np.shape(state)

        # Determine N2d and Nz
        self.meta['N2d'], self.meta['Nz'] = np.shape(self.loc['lat'])

        # Save unique forecast variable names
        self.meta['Nvars'] = len(varnames)
    

def read_parse_mpas(fnames, 
                    fix_fname, 
                    state_fields=['theta', 'qv', 'cldfrac'], 
                    other_fields=[], 
                    verbose=0):
    """
    Read and parse MPAS netCDF input

    Parameters
    ----------
    fnames : list
        NetCDF files containing MPAS atmospheric fields. Each entry is a different ensemble 
        member
    fix_fname : string
        NetCDF file containing mesh information
    state_fields : list, optional
        Fields to include in the state matrix (must be 3D)
    other_fields : list, optional
        Other fields to extract (can have any dimensions)
    verbose : int, optional
        Verbosity level

    Returns
    -------
    ens_data object
        Ensemble output

    """

    # Read in mesh info
    if verbose > 0: print('Reading MPAS mesh information')
    fix_ds = xr.open_dataset(fix_fname)
    loc = {'lat': np.rad2deg(fix_ds['latCell'].values),
           'lon': np.rad2deg(fix_ds['lonCell'].values),
           'hgt': (0.5*(fix_ds['zgrid'][:, 1:] + fix_ds['zgrid'][:, :-1]) - fix_ds['ter']).values}

    # Read in ensemble data
    if verbose > 0: print('Reading MPAS mesh atmospheric information')
    N3d = loc['hgt'].size
    Nens = len(fnames)
    state = np.zeros((N3d * len(state_fields), Nens))
    other = {}
    for f in other_fields:
        other[f] = []
    for i, f in enumerate(fnames):
        ds = xr.open_dataset(f)
        idx = 0
        for v in state_fields:
            state[idx:(idx+N3d), i] = np.flatten(ds[v].values)
            idx = idx + N3d
        for v in other_fields:
            other[v].append(np.flatten(ds[v].values))
    
    # Convert other output into arrays
    for v in other_fields:
        other[v] = np.array(other[v])

    return ens_data(state, state_fields, loc, other=other)


def read_parse_upp(fnames, 
                   state_fields=['TMP_P0_L105_GLC0', 'SPFH_P0_L105_GLC0', 'TCDC_P0_L105_GLC0'], 
                   other_fields=[], 
                   verbose=0):
    """
    Read in UPP output. 
    """


def read_ens(fnames, state_fields=[], other_fields=[], verbose=0, fix_fname=None, ftype='mpas'):
    """
    Read ensemble output and save as an ens_data object

    """

    if ftype == 'mpas':
        ens_obj = read_parse_mpas(fnames, 
                                  fix_fname, 
                                  state_fields=state_fields, 
                                  other_fields=other_fields,
                                  verbose=verbose)
    elif ftype == 'upp':
        ens_obj = read_parse_upp(fnames,
                                 state_fields=state_fields, 
                                 other_fields=other_fields,
                                 verbose=verbose)
    else:
        raise ValueError(f"ftype {ftype} is not recognized")

    return ens_obj


"""
End ens_io.py
"""