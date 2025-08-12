"""Process Level 1 IABP data for use in the distance threshold analysis"""

import numpy as np
import os
import pandas as pd
import pyproj
import sys
import ultraplot as pplt
import xarray as xr

sys.path.append("/Users/dwatkin2/Documents/research/packages/icedrift/src")

import icedrift
from icedrift import cleaning
from icedrift.utils import convert_level1_iabp

dataloc = "/Users/dwatkin2/Documents/research/data/buoy_data/IABP/level_1_data/"

def sic_along_track_monthly(position_data, sic_data, year):
    """Uses the xarray advanced interpolation to get along-track sic
    via nearest neighbors."""
    
    # Sea ice concentration uses NSIDC NP Stereographic
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
    transformer_stere = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    
    sic = pd.Series(data=np.nan, index=position_data.index)
    
    for month, group in position_data.groupby(position_data.datetime.dt.month):
        x_stere, y_stere = transformer_stere.transform(
            group.longitude, group.latitude)
        
        x = xr.DataArray(x_stere, dims="z")
        y = xr.DataArray(y_stere, dims="z")
        date = "{y}-{m}-01".format(y=year, m=month)
        SIC = sic_data.sel(time=date)['cdr_seaice_conc_monthly'].interp(
            {'x': x,
             'y': y}, method='nearest').data

        sic.loc[group.index] = np.round(SIC.T, 3)
    sic[sic >= 1] = np.nan
    return sic

columns = ['buoy_id', 'datetime', 'latitude', 'longitude', 'sic_monthly']
for year in range(2022, 2023): #2023 and 2024 are available, need to run the sic code first though
    buoy_data = []
    print(year)
    df = convert_level1_iabp(pd.read_csv(dataloc + "LEVEL1_{y}.csv".format(y=year)))
    with xr.open_dataset('../data/nsidc_sic_cdr/nsidc_cdr_sic_{y}.nc'.format(y=year)) as ds_sic:
        for buoy_id, buoy_df in df.groupby('buoy_id'):
            buoy_df['sic_monthly'] = sic_along_track_monthly(buoy_df, ds_sic, year)
            if sum(buoy_df['sic_monthly'].between(0.15, 0.8)) > 40:
                buoy_df['check_dates'] = cleaning.check_dates(buoy_df,
                                                              date_col='datetime')
                
                idx = buoy_df['check_dates'] == 0
                buoy_df['check_positions'] = False
                buoy_df.loc[idx, 'check_positions'] = cleaning.check_positions(buoy_df.loc[idx, :],
                                                                               pairs_only=True)

                idx = ~(buoy_df['check_dates'] | buoy_df['check_positions'])
                buoy_df['check_gaps'] = False
                buoy_df.loc[idx, 'check_gaps'] = cleaning.check_gaps(buoy_df.loc[idx, :],
                                                                     threshold_gap='9h',
                                                                     threshold_segment=12,
                                                                     date_col='datetime')

                idx = ~(buoy_df['check_dates'] | buoy_df['check_positions'] | buoy_df['check_gaps'])
                buoy_df['check_speed'] = False
                buoy_df.loc[idx, 'check_speed'] = cleaning.check_speed(buoy_df.loc[idx, :],
                                                                       date_index=False,
                                                                       window='3d',
                                                                       max_speed=1.5,
                                                                       date_col='datetime',
                                                                       sigma=5)
                
                buoy_df['passed_qc'] = ~(buoy_df['check_dates'] | \
                                         buoy_df['check_positions'] | \
                                         buoy_df['check_gaps'] | \
                                         buoy_df['check_speed'])
                if sum(buoy_df['passed_qc']) > 40:
                    buoy_data.append(buoy_df.loc[buoy_df['passed_qc'], columns].copy())
        # except:
        #         print(year, buoy_id, 'failed')
                
    if len(buoy_data) > 0:
        pd.concat(buoy_data).to_csv('../data/iabp_miz_data/iabp_data_{y}.csv'.format(y=year))
