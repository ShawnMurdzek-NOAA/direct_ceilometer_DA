"""
Plot Surface Stations from a BUFR File

shawn.s.murdzek@noaa.gov
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pyDA_utils.bufr as bufr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


#---------------------------------------------------------------------------------------------------
# Input Parameters
#---------------------------------------------------------------------------------------------------

bufr_fname = '/work2/noaa/wrfruc/murdzek/nature_run_winter/obs/perfect_conv_v2/real_csv/202202021100.rap.prepbufr.csv'
bufr_field = 'HOCB'
bufr_subset = 'ADPSFC'
bufr_dhr = [-0.15, 0.15]

# Options for color-coding stations (set vmin/vmax to None to use the python-computed bounds)
vmin = 0
vmax = 1000
cmap = 'plasma'

# Plotting domain [min_lon, max_lon, min_lat, max_lat]
domain = [-80, -69, 35, 43]

save_fname = 'bufr_cld_base_2022020211.png'


#---------------------------------------------------------------------------------------------------
# Create Plot
#---------------------------------------------------------------------------------------------------

bufr_csv = bufr.bufrCSV(bufr_fname)
df = bufr_csv.df.loc[(bufr_csv.df['subset'] == bufr_subset) & (bufr_csv.df['DHR'] > bufr_dhr[0]) &
                     (bufr_csv.df['DHR'] < bufr_dhr[1])]
bufr_time = dt.datetime.strptime(str(df['cycletime'].values[0]), '%Y%m%d%H')

fig = plt.figure(figsize=(10, 8))
ax = bufr.plot_obs(df, colorcode=bufr_field, scale='50m', s=50, edgecolors='k', vmin=vmin, 
                   vmax=vmax, cmap=cmap)
ax.set_extent(domain)
plt.suptitle('%s: %s $-$ %s UTC' % (bufr_subset, 
                                (bufr_time + dt.timedelta(hours=bufr_dhr[0])).strftime('%Y%m%d %H:%M'),
                                (bufr_time + dt.timedelta(hours=bufr_dhr[1])).strftime('%Y%m%d %H:%M')),
             size=16) 
plt.savefig(save_fname)


"""
End plot_sfc_stations.py
"""
