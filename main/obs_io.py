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

def read_bufr_obs(fname, subset=['ADPSFC', 'MSONET'], domain=[]):
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
    
    Returns
    -------
    pd.DataFrame
        Requested observations

    """

    # Read in BUFR CSV file
    bufr_df = bufr.bufrCSV(fname).df

    # Only retain certain subsets
    if len(subset) > 0:
        keep_idx = np.zeros(len(bufr_df))
        for s in subset:
            keep_idx[bufr_df['subset'] == s] = 1
        bufr_df = bufr_df.loc[keep_idx == 1, :].copy()
        bufr_df.reset_index(inplace=True, drop=True)

    # Remove obs outside of the desired spatial domain
    if len(domain) > 0:
        spatial_idx = np.where((bufr_df['YOB'] >= domain[0]) &
                               (bufr_df['YOB'] <= domain[2]) &
                               (bufr_df['XOB'] >= (360 + domain[1])) &
                               (bufr_df['XOB'] <= (360 + domain[3])))[0]
        bufr_df = bufr_df.iloc[spatial_idx, :]
        bufr_df.reset_index(inplace=True, drop=True)
    
    return bufr_df


"""
End obs_io.py
"""