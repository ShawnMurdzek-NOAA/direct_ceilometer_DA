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

import direct_ceilometer_DA.main.obs_io as oi


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class TestObsIO():

    @pytest.fixture(scope='class')
    def full_obs(self):
        return oi.read_bufr_obs('./sample_data/bufr/2024052704.rap.t04z.prepbufr.csv',
                                subset=[],
                                domain=[])


    def test_read_bufr_obs_subset(self, full_obs):
        df = oi.read_bufr_obs('./sample_data/bufr/2024052704.rap.t04z.prepbufr.csv',
                              subset=['ADPSFC'],
                              domain=[])
        
        assert len(full_obs) > len(df)
        assert df['subset'].unique() == ['ADPSFC']
    
    
    def test_read_bufr_obs_domain(self, full_obs):
        domain = [40, -125, 50, -100]
        df = oi.read_bufr_obs('./sample_data/bufr/2024052704.rap.t04z.prepbufr.csv',
                              subset=[],
                              domain=domain)
        
        assert len(full_obs) > len(df)
        assert df['YOB'].min() >= domain[0]
        assert df['YOB'].max() <= domain[2]
        assert df['XOB'].min() >= (domain[1] + 360)
        assert df['XOB'].max() <= (domain[3] + 360)


"""
End test_obs_io.py
"""