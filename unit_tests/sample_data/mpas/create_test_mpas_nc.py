"""
Create MPAS Test Dataset

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr


#---------------------------------------------------------------------------------------------------
# Program
#---------------------------------------------------------------------------------------------------

nens = 3
path = '/gpfs/f6/bil-pmp/scratch/Shawn.S.Murdzek/RRFSv2_runs/ens_3hrly_12km/com/rrfs/v2.0.9/rrfs.20240527/03/fcst/ens/mem{n:03d}/mpasout.2024-05-27_04.00.00.nc'
fields = ['qv', 'theta', 'cldfrac', 'qc']

for n in range(nens):
    print(f'Creating test data for member {n}')
    ds = xr.open_dataset(path.format(n=n))

    # Only keep 4 hybrid levels
    ds = ds.sel(nVertLevels=slice(0, 4))


"""
End create_test_mpas_nc.py
"""