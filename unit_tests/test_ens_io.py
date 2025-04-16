"""
Tests for ens_io.py

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pytest
import xarray as xr
import numpy as np

import direct_ceilometer_DA.main.ens_io as ei


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class TestEnsIO():

    @pytest.fixture(scope='class')
    def sample_mpas(self):
        fnames = [f'./sample_data/mpas/mem00{n}/mpasout.2024-05-27_04.00.00.TEST.nc' for n in range(1, 4)]
        state_fields = ['theta', 'qv', 'cldfrac']
        other_fields = {'qc' : 'cld_mass_mix'}
        fix_fname = './sample_data/mpas/invariant_TEST.nc'
        ftype = 'mpas'
        return ei.read_ens(fnames, 
                           state_fields=state_fields,
                           other_fields=other_fields,
                           fix_fname=fix_fname,
                           ftype=ftype)
    

    @pytest.fixture(scope='class')
    def sample_upp(self):
        fnames = [f'./sample_data/upp/mem00{n}/rrfs.t03z.natlev.TEST.f001.conus.grib2' for n in range(1, 4)]
        state_fields = ['TMP_P0_L105_GLC0', 'SPFH_P0_L105_GLC0', 'FRACCC_P0_L105_GLC0']
        other_fields = {'TKE_P0_L105_GLC0' : 'TKE'}
        ftype = 'upp'
        return ei.read_ens(fnames, 
                           state_fields=state_fields,
                           other_fields=other_fields,
                           ftype=ftype)


    def test_mpas_ens_contents(self, sample_mpas):
        """
        Check that MPAS ens_data() object has expected contents in the expected order
        """

        # Read in mesh info and a single ensemble member for comparison
        ds_info = xr.open_dataset('./sample_data/mpas/invariant_TEST.nc')
        ds = xr.open_dataset('./sample_data/mpas/mem001/mpasout.2024-05-27_04.00.00.TEST.nc')

        # Check array dimensions
        assert sample_mpas.meta['Nens'] == 3
        assert sample_mpas.meta['N2d'] == ds_info['zgrid'].shape[0]
        assert sample_mpas.meta['Nz'] == (ds_info['zgrid'].shape[1] - 1)
        assert sample_mpas.meta['Nvars'] == 3
        assert sample_mpas.meta['Nx'] == sample_mpas.meta['N2d'] * sample_mpas.meta['Nz'] * sample_mpas.meta['Nvars']


    def test_upp_ens_contents(self, sample_upp):
        """
        Check that UPP ens_data() object has expected contents in the expected order
        """

        # Read in mesh info and a single ensemble member for comparison
        ds = xr.open_dataset('./sample_data/upp/mem001/rrfs.t03z.natlev.TEST.f001.conus.grib2', engine='pynio')

        # Check array dimensions
        assert sample_upp.meta['Nens'] == 3
        assert sample_upp.meta['N2d'] == ds['gridlon_0'].size        
        assert sample_upp.meta['Nz'] == ds['HGT_P0_L105_GLC0'].shape[0]
        assert sample_upp.meta['Nvars'] == 3
        assert sample_upp.meta['Nx'] == sample_upp.meta['N2d'] * sample_upp.meta['Nz'] * sample_upp.meta['Nvars']
    

"""
End test_ens_io.py
"""