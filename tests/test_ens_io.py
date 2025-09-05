"""
Tests for ens_io.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

# Add top-level directory to PYTHONPATH
import sys
import os
path = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(path)

import pytest
import xarray as xr
import numpy as np
import copy

from main import ens_io


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class TestEnsMPASIO():

    @pytest.fixture(scope='class')
    def sample(self):
        fnames = [f'./sample_data/mpas/mem00{n}/mpasout.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]
        state_fields = ['theta', 'qv', 'cldfrac']
        other_fields = {'qc' : 'cld_mass_mix'}
        fix_fname = './sample_data/mpas/invariant_TEST.nc'
        ftype = 'mpas'
        return ens_io.read_ens(fnames, 
                           state_fields=state_fields,
                           other_fields=other_fields,
                           fix_fname=fix_fname,
                           ftype=ftype)


    def test_ens_contents(self, sample):
        """
        Check that MPAS ens_data() object has expected contents in the expected order
        """

        # Read in mesh info for comparison
        ds_info = xr.open_dataset('./sample_data/mpas/invariant_TEST.nc')

        # Check array dimensions
        assert sample.meta['Nens'] == 3
        assert sample.meta['N2d'] == ds_info['zgrid'].shape[0]
        assert sample.meta['Nz'] == (ds_info['zgrid'].shape[1] - 1)
        assert sample.meta['Nvars'] == 3
        assert sample.meta['Nx'] == sample.meta['N2d'] * sample.meta['Nz'] * sample.meta['Nvars']
    

    def test_var_dict(self, sample):
        """
        Test the .var_dict() method using MPAS netCDF output
        """
        sample = copy.deepcopy(sample)

        # Read in mesh info and netCDF output for a single member
        ds_info = xr.open_dataset('./sample_data/mpas/invariant_TEST.nc')
        ds = xr.open_dataset('./sample_data/mpas/mem001/mpasout.2024-05-27_04.00.00.TEST.nc')

        # Call method
        model_dict = sample.var_dict(0)

        # Check that atmospheric fields match
        # Account for fact that ens_io uses % for cloud fraction, whereas raw MPAS output uses decimal
        for f_origin, f_new, scale in zip(['theta', 'qv', 'cldfrac', 'qc'],
                                          ['theta', 'qv', 'cldfrac', 'cld_mass_mix'],
                                          [1, 1, 100, 1]):
            assert np.all(np.isclose(ds[f_origin].values * scale, model_dict[f_new]))
    

    def test_write_mpas_out_for_DA(self, sample):
        """
        Test the .write_mpas_out_for_DA() method using MPAS netCDF output, with output set to 0
        """
        sample = copy.deepcopy(sample)
        in_fnames = [f'./sample_data/mpas/mem00{n}/mpasout.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]
        out_fnames = [f'./sample_data/mpas/mem00{n}/mpasout.DA.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]

        # Make some changes to data, then call method
        sample.state = np.zeros(sample.state.shape)
        sample.write_mpas_out_for_DA(in_fnames, out_fnames)

        # Check to make sure output files exist
        for f in out_fnames:
            assert os.path.isfile(f)

        # Check output from a sample file
        ds_out = xr.open_dataset(out_fnames[0])
        assert np.all(np.isclose(ds_out['theta'].values, 0))
        
        # Clean up
        for f in out_fnames:
            os.remove(f)


    def test_write_mpas_out_for_DA_append(self, sample):
        """
        Test the .write_mpas_out_for_DA() method using MPAS netCDF output, with output set to 0

        This test appends to the original file rather than creating a new file
        """
        sample = copy.deepcopy(sample)
        in_fnames = [f'./sample_data/mpas/mem00{n}/mpasout.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]
        out_fnames = [f'./sample_data/mpas/mem00{n}/mpasout.DA.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]

        # Copy in_fnames to out_fnames so we don't overwrite our test data
        for in_f, out_f in zip(in_fnames, out_fnames):
            os.system(f"cp {in_f} {out_f}")

        # Check to make sure output files exist
        for f in out_fnames:
            assert os.path.isfile(f)

        # Make some changes to data, then call method
        sample.state = np.zeros(sample.state.shape)
        sample.write_mpas_out_for_DA(out_fnames, out_fnames)

        # Check output from a sample file
        ds_out = xr.open_dataset(out_fnames[0])
        assert np.all(np.isclose(ds_out['theta'].values, 0))
        
        # Clean up
        for f in out_fnames:
            os.remove(f)


    def test_write_mpas_out_for_DA_no_change(self, sample):
        """
        Test the .write_mpas_out_for_DA() method using MPAS netCDF output, but don't alter data

        Test may fail if unit conversions for cldfrac are not handled properly 
        (MPAS output uses decimals, but cloud DA code uses %)
        """
        sample = copy.deepcopy(sample)
        in_fnames = [f'./sample_data/mpas/mem00{n}/mpasout.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]
        out_fnames = [f'./sample_data/mpas/mem00{n}/mpasout.DA.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]

        # Call method
        sample.write_mpas_out_for_DA(in_fnames, out_fnames)

        # Check output from a sample file
        ds_in = xr.open_dataset(in_fnames[0])
        ds_out = xr.open_dataset(out_fnames[0])
        assert np.all(np.isclose(ds_in['cldfrac'].values, ds_out['cldfrac'].values))
        
        # Clean up
        for f in out_fnames:
            os.remove(f)


class TestEnsUPPIO():

    @pytest.fixture(scope='class')
    def sample(self):
        fnames = [f'./sample_data/upp/mem00{n}/rrfs.t03z.natlev.TEST.f001.conus.grib2' for n in range(1, 4)]
        state_fields = ['TMP_P0_L105_GLC0', 'SPFH_P0_L105_GLC0', 'FRACCC_P0_L105_GLC0']
        other_fields = {'TKE_P0_L105_GLC0' : 'TKE'}
        ftype = 'upp'
        return ens_io.read_ens(fnames, 
                               state_fields=state_fields,
                               other_fields=other_fields,
                               ftype=ftype)
    

    def test_ens_contents(self, sample):
        """
        Check that UPP ens_data() object has expected contents in the expected order
        """

        # Read a single ensemble member for comparison
        ds = xr.open_dataset('./sample_data/upp/mem001/rrfs.t03z.natlev.TEST.f001.conus.grib2', engine='pynio')

        # Check array dimensions
        assert sample.meta['Nens'] == 3
        assert sample.meta['N2d'] == ds['gridlon_0'].size        
        assert sample.meta['Nz'] == ds['HGT_P0_L105_GLC0'].shape[0]
        assert sample.meta['Nvars'] == 3
        assert sample.meta['Nx'] == sample.meta['N2d'] * sample.meta['Nz'] * sample.meta['Nvars']
    

    def test_var_dict(self, sample):
        """
        Test the .var_dict() method using UPP GRIB2 output
        """

        # Read a single ensemble member for comparison
        ds = xr.open_dataset('./sample_data/upp/mem001/rrfs.t03z.natlev.TEST.f001.conus.grib2', engine='pynio')
        shape = ds['TMP_P0_L105_GLC0'].shape

        # Call method
        model_dict = sample.var_dict(0)

        # Check that atmospheric fields match
        for f_origin, f_new in zip(['TMP_P0_L105_GLC0', 'SPFH_P0_L105_GLC0', 'FRACCC_P0_L105_GLC0', 'TKE_P0_L105_GLC0'],
                                   ['TMP_P0_L105_GLC0', 'SPFH_P0_L105_GLC0', 'FRACCC_P0_L105_GLC0', 'TKE']):
            assert np.all(np.isclose(np.reshape(ds[f_origin].values, newshape=(shape[1]*shape[2], shape[0])), 
                                     model_dict[f_new]))
    

"""
End test_ens_io.py
"""
