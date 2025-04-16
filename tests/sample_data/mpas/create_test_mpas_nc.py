"""
Create MPAS Test Dataset

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import xarray as xr


#---------------------------------------------------------------------------------------------------
# Inputs
#---------------------------------------------------------------------------------------------------

# Number of ensemble members
nens = 3

# Hybrid levels to keep
nlvl = 4

# MPAS mesh info file
info_fname = '/gpfs/f6/bil-pmp/scratch/Shawn.S.Murdzek/RRFSv2_runs/ens_3hrly_12km/stmp/20240527/rrfs_fcst_12_v2.0.9/ens/mem001/fcst_12/invariant.nc'

# Mesh info fields to save
info_fields = ['latCell', 'lonCell', 'zgrid', 'ter']

# Path to MPAS atmospheric output files (include {n} placeholder for member number)
path = '/gpfs/f6/bil-pmp/scratch/Shawn.S.Murdzek/RRFSv2_runs/ens_3hrly_12km/com/rrfs/v2.0.9/rrfs.20240527/03/fcst/ens/mem{n:03d}/mpasout.2024-05-27_04.00.00.nc'

# Atmospheric fields to save
fields = ['qv', 'theta', 'cldfrac', 'qc']


#---------------------------------------------------------------------------------------------------
# Program
#---------------------------------------------------------------------------------------------------

# Save MPAS mesh info
print('Creating netCDF file with mesh info')
ds = xr.open_dataset(info_fname)
ds = ds.sel(nVertLevels=slice(0, nlvl))
ds = ds.sel(nVertLevelsP1=slice(0, nlvl+1))
out_info = {}
for f in info_fields:
    out_info[f] = ds[f]
out_info_ds = xr.Dataset(out_info, attrs=ds.attrs)
out_info_ds.to_netcdf("./invariant_TEST.nc")

# Save MPAS atmospheric fields
for n in range(1, nens+1):
    print(f'Creating test data for member {n:03d}')
    ds = xr.open_dataset(path.format(n=n))

    # Only keep 4 hybrid levels
    ds = ds.sel(nVertLevels=slice(0, nlvl))

    # Save desired fields
    out = {}
    for f in fields:
        out[f] = ds[f]
    out_ds = xr.Dataset(out, attrs=ds.attrs)

    # Save to netCDF
    out_ds.to_netcdf(f"./mem{n:03d}/mpasout.2024-05-27_04.00.00.TEST.nc")


"""
End create_test_mpas_nc.py
"""