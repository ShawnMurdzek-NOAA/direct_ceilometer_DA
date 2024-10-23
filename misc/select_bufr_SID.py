"""
Select prepBUFR Station IDs

Command-Line Arguments
----------------------
sys.argv[1] : input prepBUFR CSV file
sys.argv[2] : output prepBUFR CSV file
sys.argv[3] : Text file where each line is a SID to retain

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import sys
import pandas as pd

from pyDA_utils import bufr


#---------------------------------------------------------------------------------------------------
# Select Observation SIDs
#---------------------------------------------------------------------------------------------------

# Inputs
in_bufr_fname = sys.argv[1]
out_bufr_fname = sys.argv[2]
txt_sid_fname = sys.argv[3]

# Open files
bufr_csv = bufr.bufrCSV(in_bufr_fname)
SIDs = pd.read_csv(txt_sid_fname, names=['SID'])

# Subset CSV
bufr_csv.select_SIDs(SIDs['SID'].values)

# Write output CSV
bufr.df_to_csv(bufr_csv.df, out_bufr_fname)


"""
End select_bufr_SID.py
"""