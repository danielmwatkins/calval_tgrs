"""Concatenate daily SIC data into one file per year.
NOTE: This script doesn't work, a version update broke it. 
This needs to get fixed."""

import xarray as xr
import os
dataloc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr_v4/'
saveloc = '../data/nsidc_sic_cdr/'

for year in range(2002, 2023):
    files = [f for f in os.listdir(dataloc + str(year)) if '.nc' in f]
    ds_year = []

    for f in files:
        with xr.open_dataset(dataloc + str(year) + '/' + f) as ds:
            ds_year.append(ds.load())
            
    xr.concat(ds_year, dim='time').to_netcdf(saveloc + '/nsidc_cdr_sic_' + str(year) + '.nc',
                                             encoding = {var:
                                                         {'zlib': True, 'complevel': 9} for var in
                                                             ['cdr_seaice_conc_monthly',
                                                              'cdr_seaice_conc_monthly_stdev',
                                                              'cdr_seaice_conc_monthly_qa_flag']})