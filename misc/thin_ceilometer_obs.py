"""
Thin Ceilometer Obs and Remove All Non-Ceilometer Obs

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import datetime as dt
import sys
import argparse
import numpy as np
import pandas as pd

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

    parser = argparse.ArgumentParser(description='Script that thins ceilometer observations and \
                                                  removes all non-ceilometer observations from a \
                                                  BUFR CSV file.')

    # Positional arguments
    parser.add_argument('in_csv',
                        help='Input BUFR CSV file',
                        type=str)

    parser.add_argument('out_csv',
                        help='Output BUFR CSV file',
                        type=str)

    parser.add_argument('min_radius',
                        help='Minimum distance between ceilometer obs (m)',
                        type=str)

    # Optional arguments
    parser.add_argument('--ADPSFC',
                        dest='only_sfc',
                        default=True,
                        help='Option to only keep ADPSFC obs',
                        type=bool)

    parser.add_argument('--limDHR',
                        dest='limDHR',
                        default=False,
                        help='Option to only retain obs closest to time DHR = 0',
                        type=bool)

    parser.add_argument('--llbox',
                        dest='apply_box',
                        default=False,
                        help='Option to subset observations into a (lat, lon) box',
                        type=bool)

    parser.add_argument('--minlat',
                        dest='minlat',
                        default=20,
                        help='Minimum latitude for (lat, lon) box (deg N)',
                        type=float)

    parser.add_argument('--minlon',
                        dest='minlon',
                        default=-120,
                        help='Minimum longitude for (lat, lon) box (deg E, 0 to 360)',
                        type=float)

    parser.add_argument('--maxlat',
                        dest='maxlat',
                        default=55,
                        help='Maximum latitude for (lat, lon) box (deg N)',
                        type=float)

    parser.add_argument('--maxlon',
                        dest='maxlon',
                        default=-60,
                        help='Maximum longitude for (lat, lon) box (deg E, 0 to 360)',
                        type=float)

    return parser.parse_args(argv)


def remove_non_ceilometer_obs(df):
    """
    Remove observations with missing ceilometer obs
    
    Code is copied from direct_ceilometer_DA/main/cloud_DA_forward_operator.py
    """

    df = df.loc[(~np.isnan(df['CLAM'])) &
                (~np.isclose(df['CLAM'], 9)) &
                (~np.isclose(df['CLAM'], 10)) &
                (~np.isclose(df['CLAM'], 14)) &
                (~np.isclose(df['CLAM'], 15)) &
                (~(~np.isclose(df['CLAM'], 0) & np.isnan(df['HOCB'])))]
    df.reset_index(inplace=True, drop=True)

    return df


if __name__ == '__main__':

    start = dt.datetime.now()
    print('Starting thin_ceilometer_obs.py')
    print(f"Time = {start.strftime('%Y%m%d %H:%M:%S')}")

    param = parse_in_args(sys.argv[1:])

    # Read in obs
    bufr_csv = bufr.bufrCSV(param.in_csv)
    print(f"Initial number of obs = {len(bufr_csv.df)}")

    # Remove non-ADPSFC obs
    if param.only_sfc:
        bufr_csv.df = bufr_csv.df.loc[bufr_csv.df['subset'] == 'ADPSFC', :]
        print(f"Number of obs after removing non-ADPSFC obs = {len(bufr_csv.df)}")

    # Only keep obs within a desired (lat, lon) box
    if param.apply_box:
        bufr_csv.select_latlon(param.minlat, param.minlon, param.maxlat, param.maxlon)
        print(f"Number of obs after applying (lat, lon) box = {len(bufr_csv.df)}")

    # Remove non-ceilometer obs
    bufr_csv.df = remove_non_ceilometer_obs(bufr_csv.df)
    print(f"Number of obs after removing non-ceilometer obs = {len(bufr_csv.df)}")

    # Only keep obs closest to DHR = 0
    # This step can be rather slow, so it's best to do it later after most of the obs have
    # already been filtered out
    if param.limDHR:
        bufr_csv.select_dhr(0)
        print(f"Number of obs after limDHR filter = {len(bufr_csv.df)}")

    # Thin obs
    thin_df = bufr.thin_obs_2d(bufr_csv.df, radius=param.min_radius, retain_all_sid=True)
    print(f"Number of obs after thinning = {len(thin_df)}")
    print(f"Number of unique sites = {len(thin_df['SID'].unique())}")

    # Write out thinned DataFrame
    bufr.df_to_csv(thin_df, param.out_csv, quotes=False)

    print('Program Finished!')
    print(f"Elapsed time = {(dt.datetime.now() - start).total_seconds()} s")


"""
End thin_ceilometer_obs.py
"""
