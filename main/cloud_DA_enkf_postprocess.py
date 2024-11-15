"""
Helper functions for post-processing EnSRF output fields

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import datetime as dt
import metpy.calc as mc
from metpy.units import units

import pyDA_utils.upp_postprocess as uppp


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def compute_RH(ens_obj, prefix=''):
    """
    Compute RH for the ensemble subset spatial domain

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    prefix : string, optional
        RH field is written to "{prefix}RH_P0_L105_GLC0" using "PRES_P0_L105_GLC0", 
        "{prefix}TMP_P0_L105_GLC0", and "{prefix}SPFH_P0_L105_GLC0"
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with the added RH fields

    """

    for mem in ens_obj.mem_names:
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"] = ens_obj.subset_ds[mem]['SPFH_P0_L105_GLC0'].copy()
        p = ens_obj.subset_ds[mem]["PRES_P0_L105_GLC0"].values * units.Pa
        T = ens_obj.subset_ds[mem][f"{prefix}TMP_P0_L105_GLC0"].values * units.K
        Q = ens_obj.subset_ds[mem][f"{prefix}SPFH_P0_L105_GLC0"].values * units.kg / units.kg
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].values = mc.relative_humidity_from_specific_humidity(p, T, Q).magnitude * 100
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].attrs['long_name'] = 'relative humidity'
        ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].attrs['units'] = '%'

    return ens_obj


def compute_pseudo_ceil(ens_obj, prefix='', RH_thres=85):
    """
    Compute pseudo ceilings based on an RH threshold for the ensemble subset spatial domain

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    prefix : string, optional
        Pseudo ceiling field is written to "{prefix}PCEIL_P0_L215_GLC0" using 
        "{prefix}RH_P0_L105_GLC0" and "HGT_P0_L105_GLC0"
    RH_thres : float, optional
        Relative humidity threshold used to determine if a pseudo ceiling exists (%)
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with the added pseudo ceiling fields (m ASL)

    """

    for mem in ens_obj.mem_names:
        ens_obj.subset_ds[mem][f"{prefix}PCEIL_P0_L215_GLC0"] = ens_obj.subset_ds[mem]['CEIL_P0_L215_GLC0'].copy()
        cond = ens_obj.subset_ds[mem][f"{prefix}RH_P0_L105_GLC0"].values >= RH_thres
        idx = np.argmax(cond, axis=0)
        shape_2d = idx.shape
        HGT_2d = ens_obj.subset_ds[mem][f"HGT_P0_L105_GLC0"].values
        PCEIL = np.zeros(shape_2d)
        for i in range(shape_2d[0]):
            for j in range(shape_2d[1]):
                PCEIL[i, j] = HGT_2d[idx[i, j], i, j]
        PCEIL[np.sum(cond, axis=0) < 1] = 2e4
        ens_obj.subset_ds[mem][f"{prefix}PCEIL_P0_L215_GLC0"].values = PCEIL
        ens_obj.subset_ds[mem][f"{prefix}PCEIL_P0_L215_GLC0"].attrs['long_name'] = f'pseudo ceilings (height where RH > {RH_thres}%)'
        ens_obj.subset_ds[mem][f"{prefix}PCEIL_P0_L215_GLC0"].attrs['units'] = 'gpm ASL'

    return ens_obj


def compute_max_lapse_rate(ens_obj, prefix=''):
    """
    Compute the maximum lapse rate for each vertical column

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    prefix : string, optional
        Max lapse rate field is written to "{prefix}MAXLR" using 
        "{prefix}TMP_P0_L105_GLC0" and "HGT_P0_L105_GLC0"
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with the added max lapse rate (K / km)

    """

    field = f"{prefix}MAXLR"
    for mem in ens_obj.mem_names:
        ens_obj.subset_ds[mem][field] = ens_obj.subset_ds[mem]['CEIL_P0_L215_GLC0'].copy()
        T3D = ens_obj.subset_ds[mem][f"{prefix}TMP_P0_L105_GLC0"].values
        Z3D = ens_obj.subset_ds[mem][f"HGT_P0_L105_GLC0"].values
        ens_obj.subset_ds[mem][field].values = np.amax((T3D[:-1, :, :] - T3D[1:, :, :]) / 
                                                       (1e-3*(Z3D[1:, :, :] - Z3D[:-1, :, :])), axis=0)
        ens_obj.subset_ds[mem][field].attrs['long_name'] = 'column-maximum lapse rate'
        ens_obj.subset_ds[mem][field].attrs['units'] = 'K / km'

    return ens_obj


def convert_ceil_agl(ens_obj, compute_ceil_kw={}):
    """
    Convert cloud ceilings from height ASL to height AGL

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    compute_ceil_kw : dictionary, optional
        Keyword arguments passed to uppp.compute_ceil_agl()
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with cloud ceilings in height AGL

    """

    for mem in ens_obj.mem_names:
        ens_obj.subset_ds[mem] = uppp.compute_ceil_agl(ens_obj.subset_ds[mem], **compute_ceil_kw)   

    return ens_obj


def compute_ens_stats_3D(ens_obj, field, stat_fct={'mean_':np.mean, 'std_': np.std}):
    """
    Compute various ensemble stats for a 3D field that is not part of the state vector

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    field : string
        Field from ens_obj.subset_ds[m] to compute statistics for
    stat_fct : dictionary, optional
        Statistics to compute. Key is the prefix added to the resulting field and the value is the 
        function
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with statistics for the desired field added to the first ensemble member

    Notes
    -----
    Output is saved to the first ensemble member

    """

    mem_names = ens_obj.mem_names
    shape = ens_obj.subset_ds[mem_names[0]][field].shape
    full_array = np.zeros([shape[0], shape[1], shape[2], len(mem_names)])

    # Extract the desired field from each ensemble member
    for i, m in enumerate(mem_names):
        full_array[:, :, :, i] = ens_obj.subset_ds[m][field].values
    
    # Compute stats
    for f in stat_fct.keys():
        ens_obj.subset_ds[mem_names[0]][f"{f}{field}"] = ens_obj.subset_ds[mem_names[0]][field].copy()
        ens_obj.subset_ds[mem_names[0]][f"{f}{field}"].values = stat_fct[f](full_array, axis=3)

    return ens_obj


def compute_ens_incr_3D(ens_obj, field):
    """
    Compute analysis increments for the desired field

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    field : string
        Field from ens_obj.subset_ds[m] to compute statistics for
    
    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output with analysis increments (using the prefix "incr_")

    Notes
    -----
    The mean analysis increment is saved to the first ensemble member with the prefix "mean_incr_"

    """

    mem_names = ens_obj.mem_names
    incr_sum = np.zeros(ens_obj.subset_ds[mem_names[0]][f"bgd_{field}"].shape)

    # Compute increment for each ensemble member
    for m in mem_names:
        ens_obj.subset_ds[m][f"incr_{field}"] = ens_obj.subset_ds[m][f"bgd_{field}"].copy()
        ens_obj.subset_ds[m][f"incr_{field}"].values = ens_obj.subset_ds[m][f"ana_{field}"].values - ens_obj.subset_ds[m][f"bgd_{field}"].values
        incr_sum = incr_sum + ens_obj.subset_ds[m][f"incr_{field}"].values
    
    # Add average increment to first ensemble member
    ens_obj.subset_ds[mem_names[0]][f"mean_incr_{field}"] = ens_obj.subset_ds[mem_names[0]][f"bgd_{field}"].copy()
    ens_obj.subset_ds[mem_names[0]][f"mean_incr_{field}"].values = incr_sum / len(mem_names)

    return ens_obj


def post_enkf(ens_obj, param, DA=True, pseudo_ceil=True, max_lr=True, compute_ceil_agl=True):
    """
    Main driver for postprocessing EnKF output

    * Compute derived fields (RH, pseudo ceilings, max lapse rate)
    * Create analysis (ana_) fields
    * Compute ensemble statistics
    * Compute analysis increments

    Parameters
    ----------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output
    param : dictionary
        YAML inputs
    DA : boolean, optional
        Was DA actually performed?
    pseudo_ceil : boolean, optional
        Option to compute pseudo ceilings using RH
    max_lr : boolean, optional
        Option to compute column-maximum lapse rates
    compute_ceil_agl : boolean, optional
        Option to compute cloud ceiling AGL heights

    Returns
    -------
    ens_obj : pyDA_utils.ensemble_utils.ensemble object
        Ensemble output

    """

    # Save analysis fields
    prefixes = ['bgd_']
    if DA:
        for ens in ens_obj.mem_names:
            for v in param['state_vars']:
                ens_obj.subset_ds[ens]['ana_'+v] = ens_obj.subset_ds[ens][v].copy()
        prefixes.append('ana_')

    # Cloud ceiling fields to convert to AGL
    ceil_fields = {'CEIL_LEGACY':'HGT_P0_L215_GLC0', 
                   'CEIL_EXP1':'CEIL_P0_L215_GLC0',
                   'CEIL_EXP2':'CEIL_P0_L2_GLC0'}
    
    # Fields to compute increments and statistics for
    incr_fields = param['state_vars'] + ['RH_P0_L105_GLC0']
    stats_fields = param['state_vars'] + ['RH_P0_L105_GLC0']

    # Compute derived fields
    for p in prefixes:
        ens_obj = compute_RH(ens_obj, prefix=p)
        if pseudo_ceil:
            ens_obj = compute_pseudo_ceil(ens_obj, prefix=p, RH_thres=param['plot_postage_config']['pseudo_ceil_RH_thres'])
            ceil_fields[f'{p}PCEIL'] = f'{p}PCEIL_P0_L215_GLC0'
            incr_fields.append('PCEIL')
        if max_lr:
            ens_obj = compute_max_lapse_rate(ens_obj, prefix=p)
            incr_fields.append('MAXLR')
    
    # Compute cloud ceiling heights AGL
    if compute_ceil_agl:
        ens_obj = convert_ceil_agl(ens_obj, compute_ceil_kw={'no_ceil':np.nan, 'fields':ceil_fields})
    
    # Compute ensemble stats and analysis increment
    for v in stats_fields:
        for p in prefixes:
            ens_obj = compute_ens_stats_3D(ens_obj, f"{p}{v}")
    if DA:
        for v in incr_fields:
            ens_obj = compute_ens_incr_3D(ens_obj, v)
    
    return ens_obj


"""
End cloud_DA_enkf_postprocess.py
"""