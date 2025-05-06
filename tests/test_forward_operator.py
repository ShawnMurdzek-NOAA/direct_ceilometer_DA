"""
Tests for cloud_DA_forward_operator.py

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
from haversine import haversine_vector

from main import ens_io
from main import obs_io
import main.cloud_DA_forward_operator as cfo


#---------------------------------------------------------------------------------------------------
# Contents
#---------------------------------------------------------------------------------------------------

class TestForwardOperator():

    @pytest.fixture(scope='class')
    def raw_sample(self):

        # Model data
        fnames = ['./sample_data/mpas/mem001/mpasout.2024-05-27_04.00.00.TEST.nc']
        state_fields = ['cldfrac']
        other_fields = {}
        fix_fname = './sample_data/mpas/invariant_TEST.nc'
        ftype = 'mpas'
        ens_obj = ens_io.read_ens(fnames, 
                                  state_fields=state_fields,
                                  other_fields=other_fields,
                                  fix_fname=fix_fname,
                                  ftype=ftype)
        model_dict = ens_obj.var_dict(0)

        # Ceilometer data
        cld_df = obs_io.read_bufr_obs('./sample_data/bufr/2024052704.rap.t04z.prepbufr.csv',
                                      subset=['ADPSFC'],
                                      domain=[])

        return model_dict, cld_df


    def test_remove_missing_cld_ob(self, raw_sample):
        cld_df = copy.deepcopy(raw_sample[1])

        # Add obs that should be removed
        cld_df.loc[cld_df['SID'] == 'KHFD', 'CLAM'] = np.nan
        cld_df.loc[cld_df['SID'] == 'KBOS', 'CLAM'] = 9
        cld_df.loc[cld_df['SID'] == 'KBDL', 'CLAM'] = 10
        cld_df.loc[cld_df['SID'] == 'KDEN', 'CLAM'] = 14
        cld_df.loc[cld_df['SID'] == 'KPHX', 'CLAM'] = 15
        cld_df.loc[cld_df['SID'] == 'KJFK', 'CLAM'] = 5
        cld_df.loc[cld_df['SID'] == 'KJFK', 'HOCB'] = np.nan

        out_df = cfo.remove_missing_cld_ob(cld_df)

        # Check that desired obs have been removed
        for code in [9, 10, 14, 15]:
            assert np.sum(out_df['CLAM'] == code) == 0
        assert np.sum(np.isnan(out_df['CLAM'])) == 0
        assert np.sum(np.logical_and(np.isnan(out_df['HOCB']), (out_df['CLAM'] != 0))) == 0

    
    def test_ceilometer_hofx_driver(self, raw_sample):
        raw_sample = copy.deepcopy(raw_sample)
        cld_df = cfo.remove_missing_cld_ob(raw_sample[1])

        cld_hofx = cfo.ceilometer_hofx_driver(cld_df, raw_sample[0], cld_field='cldfrac', verbose=1)

        # Just check to see whether 'hofx' is in the output (i.e., we are mainly just seeing 
        # whether ceilometer_hofx_driver throws an error). Additional tests below will test the
        # different pieces of cfo.ceilometer_hofx_driver()
        assert 'hofx' in cld_hofx.data
    

    @pytest.fixture(scope='class')
    def sample(self, raw_sample):
        cld_df = cfo.remove_missing_cld_ob(raw_sample[1])
        return cfo.sfc_cld_forward_operator(cld_df, raw_sample[0], cld_field='cldfrac')
    

    def test_decode_ob_clam(self, sample):
        hofx = copy.deepcopy(sample)
        hofx.decode_ob_clam()

        # Check that the proper fields were added
        for field in ['ob_cld_amt', 'ob_cld_precision']:
            assert field in hofx.data
        
        # Check that the values are within the acceptable set of values
        for i in hofx.data['idx']:
            for j in range(len(hofx.data['CLAM'][i])):
                assert hofx.data['ob_cld_amt'][i][j] in [0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
                assert hofx.data['ob_cld_precision'][i][j] in [12.5, 25]


    def test_interp_model_col_to_ob_nearest(self, sample):
        hofx = copy.deepcopy(sample)
        hofx.interp_model_col_to_ob(method='nearest')

        # Check that the proper fields were added
        for field in ['x_proj', 'y_proj']:
            assert field in hofx.model_dict
        for field in ['x_proj', 'y_proj', 'model_zgrid', 'model_col_cldfrac', 'model_col_hgt']:
            assert field in hofx.data

        # Spot check: Closest (lat, lon) coordinate to KHFD (Hartford, CT)
        i = np.where(hofx.data['SID'] == 'KHFD')[0][0]
        ob_pt = np.array([hofx.data['lat'][i], hofx.data['lon'][i]]).T
        model_pts = np.array([hofx.model_dict['lat'], hofx.model_dict['lon']]).T
        dist = np.squeeze(haversine_vector(model_pts, ob_pt, check=False, comb=True))
        i_model = np.argmin(dist)
        assert np.all(np.isclose(hofx.data['model_col_cldfrac'][i], hofx.model_dict['cldfrac'][i_model, :]))
        assert np.all(np.isclose(hofx.data['model_col_hgt'][i], hofx.model_dict['hgt'][i_model, :]))
        assert np.all(np.isclose(hofx.data['model_zgrid'][i], np.array([0, 1, 2, 3])))

    
    def test_impose_hgt_limits(self, sample):
        hofx = copy.deepcopy(sample)
        hofx.interp_model_col_to_ob(method='nearest')

        # Add data to be removed
        hofx.data['model_col_hgt'][0][0] = -5
        hofx.data['model_col_hgt'][1][-1] = 4000

        hofx.impose_hgt_limits(min_hgt=10, max_hgt=3000, hgt_field='model_col_hgt', 
                               fields=['model_col_hgt', 'model_col_cldfrac', 'model_zgrid'])
        
        # Check that height limits are not exceeded
        for i in hofx.data['idx']:
            assert max(hofx.data['model_col_hgt'][i]) <= 3000
            assert min(hofx.data['model_col_hgt'][i]) >= 10


    def test_impose_min_cld_frac(self, sample):
        hofx = copy.deepcopy(sample)
        hofx.interp_model_col_to_ob(method='nearest')

        # Add data to be removed
        hofx.data['model_col_cldfrac'][0][0] = 10

        hofx.impose_min_cld_frac(min_cld_frac=20, field='model_col_cldfrac')
        
        # Check that cloud fractions are in the desired range
        for i in hofx.data['idx']:
            for frac in hofx.data['model_col_cldfrac'][i]:
                assert (frac > 20) or (frac == 0)


    def test_interp_ob_hgt_to_model_grid(self, sample):
        hofx = copy.deepcopy(sample)
        hofx.interp_model_col_to_ob(method='nearest')
        hofx.interp_ob_hgt_to_model_grid()

        # Check that the proper fields are added
        assert 'ob_hgt_model' in hofx.data

    
    def test_clean_obs(self, sample):
        hofx = copy.deepcopy(sample)

        # Create data to be removed
        sid = hofx.data['SID'][0]
        for i in np.where(hofx.data['SID'] == sid)[0]:
            hofx.data['HOCB'][i] = []
        
        hofx.clean_obs()

        assert sid not in hofx.data['SID']

    
    def test_add_clear_obs(self, sample):
        hofx = copy.deepcopy(sample)
        hofx.decode_ob_clam()

        # Edit a specific ob
        hofx.data['ob_cld_amt'][0] = [75, 100]
        hofx.data['HOCB'][0] = [100, 140]
        hofx.data['CLAM'][0] = [6, 8]

        hofx.add_clear_obs(clr_ob_locs=np.array([30, 60, 90, 120, 150, 180]))

        # Check that the ob we edited had the proper clear obs added to it
        assert np.all(np.isclose(hofx.data['ob_cld_amt'][0], np.array([0, 0, 75, 0, 100])))
        assert np.all(np.isclose(hofx.data['HOCB'][0], np.array([30, 60, 100, 120, 140])))
    

    def test_interp_model_to_obs(self, sample):
        hofx = copy.deepcopy(sample)
        
        # Create fake data
        hofx.data = {'idx': [0, 1],
                     'model_col_hgt': [[10, 60.2, 130, 155], [11, 63.1, 122.4, 160]],
                     'model_col_cldfrac': [[0, 0, 25, 75], [0, 25, 20, 0]],
                     'HOCB': [[80, 152.3], [130]],
                     'ob_cld_amt': [[25, 50], [25]],
                     'ob_cld_precision': [[12.5, 12.5], [25]]}


        hofx.interp_model_to_obs(method='nearest', match_precision=True, field='model_col_cldfrac')

        # Check results
        assert np.all(np.isclose(hofx.data['hofx'][0], np.array([0, 75])))
        assert np.all(np.isclose(hofx.data['hofx'][1], np.array([25])))


"""
End test_forward_operator.py
"""
