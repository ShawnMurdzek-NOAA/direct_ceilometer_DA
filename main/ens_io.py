"""
Functions for Cloud DA Ensemble I/O

All necessary unit conversions for ensemble data are also handled here

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import numpy as np


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class ens_data():
    """
    Class to handle ensemble output for cloud DA. Note that cloud fraction must be in %, not decimal.

    Parameters
    ----------
    state : np.array
        State matrix used for DA. Dimensions: (Nx, Nens)
    varnames : np.array
        Names of the forecast variables in state. Dimensions: (Nvars)
    loc : dictionary
        Location of forecast variables. Must include:
            lat : Latitude in deg N. Dimensions: (N2d)
            lon : Longitude in deg E and in range (-180, 180). Dimensions: (N2d)
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
        self.meta['N2d'], self.meta['Nz'] = np.shape(self.loc['hgt'])

        # Save unique forecast variable names
        self.meta['Nvars'] = len(varnames)
    

    def var_dict(self, n):
        """
        Return all variables (in state, loc, and other) as a dictionary for a single ensemble member

        Parameters
        ----------
        n : integer
            Ensemble member number (starting with 0)
        
        Returns
        -------
        dictionary
            Model output as a dictionary

        """
        
        out_dict = {}
        N2d = self.meta['N2d']
        Nz = self.meta['Nz']
        N3d = N2d * Nz

        # State variables
        for i, v in enumerate(self.varnames):
            out_dict[v] = np.reshape(self.state[(N3d*i):(N3d*(i+1)), n], newshape=(N2d, Nz))

        # Locations
        for key in self.loc.keys():
            out_dict[key] = self.loc[key]

        # Other variables
        for key in self.other.keys():
            out_dict[key] = np.reshape(self.other[key][:, n], newshape=(N2d, Nz))

        return out_dict
    

    def write_mpas_out_for_DA(self, in_fnames, out_fnames):
        """
        Write ensemble output to an MPAS netCDF file

        Parameters
        ----------
        in_fnames : list
            List of input MPAS file names. Dimensions: (Nens)
        out_fnames : list
            List of output MPAS file names. Dimensions: (Nens)
        
        Returns
        -------
        None

        """

        for i, (in_f, out_f) in enumerate(zip(in_fnames, out_fnames)):
            if (in_f == out_f):
                ds = xr.open_dataset(in_f, mode='a')
            else:
                ds = xr.open_dataset(in_f)
            model_dict = self.var_dict(i)
            for v in self.varnames:
                data = model_dict[v]
                if v == 'cldfrac': data = data * 0.01
                ds[v].values = np.expand_dims(data, axis=0)
            if (in_f == out_f):
                ds.to_netcdf(out_f, mode='a')
            else:
                ds.to_netcdf(out_f)
    

def read_parse_mpas(fnames, 
                    fix_fname, 
                    state_fields=['theta', 'qv', 'cldfrac'], 
                    other_fields={}, 
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
    other_fields : dictionary, optional
        Other fields to extract (can have any dimensions). Key is MPAS field name, value is the 
        general name for the field
    verbose : int, optional
        Verbosity level

    Returns
    -------
    ens_data object
        Ensemble output

    """

    # Read in mesh info
    if verbose > 0: print('  Reading MPAS mesh information')
    fix_ds = xr.open_dataset(fix_fname)
    loc = {'lat': np.rad2deg(fix_ds['latCell'].values),
           'lon': np.rad2deg(fix_ds['lonCell'].values) - 360,
           'hgt': (0.5*(fix_ds['zgrid'][:, 1:] + fix_ds['zgrid'][:, :-1]) - fix_ds['ter']).values}

    # Read in ensemble data
    if verbose > 0: print('  Reading MPAS mesh atmospheric information')
    N3d = loc['hgt'].size
    Nens = len(fnames)
    state = np.zeros((N3d * len(state_fields), Nens))
    other = {}
    for key in other_fields.keys():
        other[other_fields[key]] = []
    for i, f in enumerate(fnames):
        ds = xr.open_dataset(f)
        idx = 0
        for v in state_fields:
            if v == 'cldfrac':
                data = ds[v].values * 100
            else:
                data = ds[v].values
            state[idx:(idx+N3d), i] = np.ravel(data)
            idx = idx + N3d
        for key in other_fields.keys():
            other[other_fields[key]].append(np.ravel(ds[key].values))
    
    # Convert other output into arrays
    for key in other_fields.keys():
        name = other_fields[key]
        other[name] = np.array(other[name]).T

    return ens_data(state, state_fields, loc, other=other)


def read_parse_upp(fnames, 
                   state_fields=['TMP_P0_L105_GLC0', 'SPFH_P0_L105_GLC0', 'FRACCC_P0_L105_GLC0'], 
                   other_fields={}, 
                   verbose=0):
    """
    Read and parse UPP GRIB2 input

    Parameters
    ----------
    fnames : list
        GRIB2 files containing UPP output. Each entry is a different ensemble member
    state_fields : list, optional
        Fields to include in the state matrix (must be 3D)
    other_fields : dictionary, optional
        Other fields to extract (can have any dimensions). Key is UPP field name, value is the 
        general name for the field
    verbose : int, optional
        Verbosity level

    Returns
    -------
    ens_data object
        Ensemble output

    """

    # Read in grid info
    if verbose > 0: print('  Reading UPP grid information')
    fix_ds = xr.open_dataset(fnames[0], engine='pynio')
    shape_3d = fix_ds['HGT_P0_L105_GLC0'].shape
    loc = {'lat': np.ravel(fix_ds['gridlat_0'].values),
           'lon': np.ravel(fix_ds['gridlon_0'].values),
           'hgt': np.reshape(fix_ds['HGT_P0_L105_GLC0'].values - 
                             fix_ds['HGT_P0_L1_GLC0'].values[np.newaxis, :, :], 
                             newshape=(shape_3d[1] * shape_3d[2], shape_3d[0]))}

    # Read in ensemble data
    if verbose > 0: print('  Reading UPP atmospheric information')
    N3d = loc['hgt'].size
    Nens = len(fnames)
    state = np.zeros((N3d * len(state_fields), Nens))
    other = {}
    for key in other_fields.keys():
        other[other_fields[key]] = []
    for i, f in enumerate(fnames):
        ds = xr.open_dataset(f, engine='pynio')
        idx = 0
        for v in state_fields:
            state[idx:(idx+N3d), i] = np.ravel(ds[v].values)
            idx = idx + N3d
        for key in other_fields.keys():
            other[other_fields[key]].append(np.ravel(ds[key].values))
    
    # Convert other output into arrays
    for key in other_fields.keys():
        name = other_fields[key]
        other[name] = np.array(other[name]).T

    return ens_data(state, state_fields, loc, other=other)


def read_ens(fnames, 
             state_fields=['theta', 'qv', 'cldfrac'], 
             other_fields={}, 
             verbose=0, 
             fix_fname=None, 
             ftype='mpas'):
    """
    Read ensemble output

    Parameters
    ----------
    fnames : list
        Ensemble member file names
    state_fields : list, optional
        Fields to include in the state matrix (must be 3D)
    other_fields : dictionary, optional
        Other fields to extract (can have any dimensions). Key is field name, value is the 
        general name for the field
    verbose : int, optional
        Verbosity level
    fix_fname : string, optional
        File containing grid or mesh information. Only needed for MPAS output
    ftype : string, optional
        Input file type. Options: 'mpas' or 'upp'
    
    Returns
    -------
    ens_data object
        Ensemble output

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
