"""
Functions for Cloud DA BUFR Observation I/O

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr
import numpy as np
import pandas as pd

from pyDA_utils import bufr


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

def read_bufr_obs(fname, subset=['ADPSFC', 'MSONET'], domain=[], lim_DHR=True, verbose=0):
    """
    Read BUFR CSV observations

    Parameters
    ----------
    fname : string
        BUFR CSV file name
    subset : list, optional
        Observation subsets to retain. Set to an empty list to not use
    domain : list, optional
        Only keep obs within the specified spatial domain, [minlat, minlon, maxlat, maxlon].
        Set to an empty list to not use
    lim_DHR : boolean, optional
        If multiple obs for a single SID, option to only keep the ob with a DHR closest to 0
    verbose : integer, optional
        Verbosity level
    
    Returns
    -------
    pd.DataFrame
        Requested observations

    """

    # Read in BUFR CSV file
    bufr_csv = bufr.bufrCSV(fname)

    # Only retain certain subsets
    if len(subset) > 0:
        keep_idx = np.zeros(len(bufr_csv.df))
        for s in subset:
            keep_idx[bufr_csv.df['subset'] == s] = 1
        bufr_csv.df = bufr_csv.df.loc[keep_idx == 1, :].copy()
        bufr_csv.df.reset_index(inplace=True, drop=True)

    # Remove obs outside of the desired spatial domain
    if len(domain) > 0:
        spatial_idx = np.where((bufr_csv.df['YOB'] >= domain[0]) &
                               (bufr_csv.df['YOB'] <= domain[2]) &
                               (bufr_csv.df['XOB'] >= (360 + domain[1])) &
                               (bufr_csv.df['XOB'] <= (360 + domain[3])))[0]
        bufr_csv.df = bufr_csv.df.iloc[spatial_idx, :]
        bufr_csv.df.reset_index(inplace=True, drop=True)
    
    # Only retain observations closest to DHR = 0
    if lim_DHR: bufr_csv.select_dhr(0)

    if verbose: print(f"  Total number of obs = {len(bufr_csv.df)}")
    
    return bufr_csv.df


"""
End obs_io.py
"""