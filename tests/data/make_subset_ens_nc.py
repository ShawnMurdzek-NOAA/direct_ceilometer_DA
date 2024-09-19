"""
Make Ensemble Subset NetCDF Data for Testing

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pyDA_utils.ensemble_utils as eu
import direct_ceilometer_DA.drivers.probing_rrfs_ensemble as pre
import yaml
import datetime as dt


#---------------------------------------------------------------------------------------------------
# Create Pickled Data
#---------------------------------------------------------------------------------------------------

yml_fname = 'create_test_data.yml'

# Read in sample data
with open(yml_fname, 'r') as fptr:
    param = yaml.safe_load(fptr)
ens_obj = pre.read_ensemble_output(param)

# Create netCDF
start = dt.datetime.now()
ens_obj.save_subset_ens('test_data.nc')
print(f"time to save output = {(dt.datetime.now() - start).total_seconds()} s")


"""
End make_ens_pickle.py
"""
