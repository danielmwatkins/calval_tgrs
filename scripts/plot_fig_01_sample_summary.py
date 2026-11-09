import pandas as pd
import ultraplot as uplt
import cartopy.crs as ccrs
import numpy as np
import warnings
import xarray as xr
import pyproj
import os

warnings.simplefilter('ignore')
dataloc = '../../ice_floe_validation_dataset/'

uplt.rc['reso'] = 'med'
uplt.rc['cartopy.circular'] = False

# Load region data
regions = pd.read_csv(dataloc + 'data/metadata/region_definitions.csv', index_col=0)
regions['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in regions.index]

# Put in longitude order starting with Greenland
regions = regions.loc[['greenland_sea',
                       'barents_kara_seas',
                       'laptev_sea',
                       'sea_of_okhostk',
                       'east_siberian_sea',
                       'bering_chukchi_seas',
                       'beaufort_sea',
                       'hudson_bay',
                       'baffin_bay'],:]

crs0 = pyproj.CRS('WGS84')
crs1 = pyproj.CRS('epsg:3413')
transformer_xy_ll = pyproj.Transformer.from_crs(crs1, crs_to=crs0, always_xy=True)
case_list = pd.read_csv(dataloc + 'data/metadata/validation-region-case-100km.csv')
case_list_500_km = case_list.copy()
dx = 250e3
for row, data in case_list_500_km.iterrows():
    data['left_x'] = data['center_x'] - dx
    data['right_x'] = data['center_x'] + dx
    data['lower_y'] = data['center_y'] - dx
    data['upper_y'] = data['center_y'] + dx
    region_left = regions.loc[data['region_name'], 'left_x']
    region_right = regions.loc[data['region_name'], 'right_x']
    region_upper = regions.loc[data['region_name'], 'upper_y']
    region_lower = regions.loc[data['region_name'], 'lower_y']

    if region_left > data['left_x']:
        adj_dx = region_left - data['left_x']
        data['left_x'] += adj_dx
        data['right_x'] += adj_dx
        data['center_x'] += adj_dx
        
    elif region_right < data['right_x']:
        adj_dx = data['right_x'] - region_right
        data['left_x'] -= adj_dx
        data['right_x'] -= adj_dx
        data['center_x'] -= adj_dx

    if region_lower > data['lower_y']:
        adj_dy = region_lower - data['lower_y']
        data['lower_y'] += adj_dy
        data['upper_y'] += adj_dy
        data['center_y'] += adj_dy
        
    elif region_upper < data['upper_y']:
        adj_dy = data['upper_y'] - region_upper
        data['lower_y'] -= adj_dy
        data['upper_y'] -= adj_dy
        data['center_y'] -= adj_dy
    
    case_list_500_km.loc[row, 'left_x'] = data['left_x']
    case_list_500_km.loc[row, 'right_x'] = data['right_x']
    case_list_500_km.loc[row, 'lower_y'] = data['lower_y']
    case_list_500_km.loc[row, 'upper_y'] = data['upper_y']
    case_list_500_km.loc[row, 'center_x'] = data['center_x']
    case_list_500_km.loc[row, 'center_y'] = data['center_y']

lon, lat = transformer_xy_ll.transform(case_list_500_km.center_x, case_list_500_km.center_y)
case_list_500_km['center_lon'] = lon.round(3)
case_list_500_km['center_lat'] = lat.round(3)



# Load SIC mean data
ds_april_mean = xr.open_dataset(dataloc + 'data/nsidc/mean_sic_april_2000-2024.nc')
ds_sept_mean = xr.open_dataset(dataloc + 'data/nsidc/mean_sic_september_2000-2024.nc')
X = ds_april_mean.x
Y = ds_sept_mean.y


df = pd.read_csv(dataloc + 'data/validation_dataset/validation_dataset.csv', parse_dates=['start_date'])
df['month'] = df['start_date'].dt.month
df = df.loc[df.satellite=='aqua']
df.index = [str(x).zfill(3) for x in df.case_number]

crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
layout = [[1, 2, 5],
          [1, 3, 6],
          [1, 4, 7]]
fig, axs = uplt.subplots(layout, share=False,
                         width=12, height=8,
                         wratios=[3, 2, 1],
                         proj={1: 'npstere'}, proj_kw={'lon_0': -45})
ax = axs[0,0]
ax.format(land=True, color='k', boundinglat=52, landzorder=0, latmax=90, facecolor='w', lonlocator=10)
ax.set_extent([-3.5e6, 2.8e6, -4e6, 4.4e6], crs=crs)
    
    
colors = {region: c['color'] for region, c in zip(
            regions.index,
            uplt.Cycle('colorblind10', len(regions)))}
linestyles = {region: ls for region, ls in zip(regions.index,
                        ['-', '-.', '--', '-', '-.', '--', '-.', '-', '--'])}

regions['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in regions.index]
regions = regions.sort_values('center_lon')

for idx, row in regions.iterrows():
    if row.print_title == 'Barents Kara Seas':
        regions.loc[idx, 'print_title'] = 'Barents-Kara Seas'
    if row.print_title == 'Bering Chukchi Seas':
        regions.loc[idx, 'print_title'] = 'Bering-Chukchi Seas'


miz = (ds_april_mean['cdr_seaice_conc_monthly'] > 0.15) & \
            (ds_april_mean['cdr_seaice_conc_monthly'] < 0.85)
april_miz = np.ma.masked_array(np.ones(miz.data.shape), mask=~miz)    

miz = (ds_sept_mean['cdr_seaice_conc_monthly'] > 0.15) &  \
            (ds_sept_mean['cdr_seaice_conc_monthly'] < 0.85)
sept_miz = np.ma.masked_array(np.ones(miz.data.shape), mask=~miz)    

apr_pack_ice = (ds_april_mean['cdr_seaice_conc_monthly'] >= 0.85) & \
            (ds_april_mean['cdr_seaice_conc_monthly'] <= 1)
apr_pack_ice = np.ma.masked_array(np.ones(apr_pack_ice.data.shape), mask=~apr_pack_ice)    

sep_pack_ice = (ds_sept_mean['cdr_seaice_conc_monthly'] >= 0.85) & \
            (ds_sept_mean['cdr_seaice_conc_monthly'] <= 1)

sep_pack_ice = np.ma.masked_array(np.ones(sep_pack_ice.data.shape), mask=~sep_pack_ice)    


ax.pcolormesh(X, Y, apr_pack_ice, vmin=0, vmax=1, color='blue1', alpha=1,
              transform=crs, label='')
ax.pcolormesh(X, Y, april_miz, vmin=0, vmax=1, color='blue4', alpha=1,
              transform=crs, label='')
ax.pcolormesh(X, Y, sep_pack_ice, vmin=0, vmax=1, color='orange1', alpha=1,
              transform=crs, label='')
ax.pcolormesh(X, Y, sept_miz, vmin=0, vmax=1, color='orange3', alpha=1,
              transform=crs, label='September MIZ')

h = [ax.plot([],[],marker='s', lw=0, color=color) for color in ['blue4', 'blue1', 'orange3', 'orange1']]
ax.legend(h, ['April MIZ', 'April Pack Ice', 'September MIZ', 'September Pack Ice'], loc='ur', ncols=1, alpha=1)


idx = 1
for idx, region, lat, lon in zip(range(len(regions)), regions.index, regions.center_lat, regions.center_lon):

    xbox = np.array(regions.loc[region, ['left_x', 'left_x', 'right_x', 'right_x', 'left_x']].astype(float))
    ybox = np.array(regions.loc[region, ['lower_y', 'upper_y', 'upper_y', 'lower_y', 'lower_y']].astype(float))
    
    ax.plot(xbox, ybox, transform=crs,
            label='({n}) {t}'.format(n=idx + 1, t=regions.loc[region, 'print_title']), 
               color=colors[region], ls=linestyles[region], m='', zorder=5, lw=3)

ice_floes_cases = df.loc[df.visible_floes == 'yes']
no_floes_cases = df.loc[df.visible_floes == 'no']


ax.set_extent([-3.5e6, 2.8e6, -4e6, 4.4e6], crs=crs)
box_cases = [11, 108, 166]
ax.scatter(df.center_x, df.center_y, transform=crs, facecolor='w', edgewidth=1, edgecolor='k', marker='o', ms=15)

for idx, case in df.iterrows():
    if case.case_number in box_cases:
        x0 = case.center_x
        y0 = case.center_y
        left = x0 - 50e3
        right = x0 + 50e3
        bottom = y0 - 50e3
        top = y0 + 50e3 
        region = case.region
        c = 'r'
        z = 20
        ax.plot([left, left, right, right, left],
            [bottom, top, top, bottom, bottom], label='',
                transform=crs, color=c, lw=1, zorder=z)
    
for idx, case in case_list_500_km.iterrows():
    if case.case_number in box_cases:
        x0 = case.center_x
        y0 = case.center_y
        left = case.left_x
        right = case.right_x
        bottom = case.lower_y
        top = case.upper_y
        region = case.region
        c = 'r'
        z = 25
        ax.plot([left, left, right, right, left],
            [bottom, top, top, bottom, bottom], label='',
                transform=crs, color=c, lw=1, zorder=z)
   
ax.plot([],[],m='s', color='gray', label='No visible floes', lw=0)

regions["idx_num"] = np.arange(1, len(regions) + 1)
for region in regions.index:
    ax.text(regions.loc[region, 'left_x'] + 150e3,
            regions.loc[region, 'upper_y'] - 200e3, str(regions.loc[region, "idx_num"]),
            transform=crs, bbox=True, bboxalpha=1,
            border=False, color='k', borderwidth=0,
            bboxstyle='circle', bboxcolor='w', zorder=30)
ax.text(100, 69, 'Arctic Circle', color='lightgray', transform=ccrs.PlateCarree(), rotation=-40)
ax.format(title='Location of Sampled Scenes')
# fig.legend(ncols=3, alpha=1, loc='b', order='F')
ax.plot(np.linspace(0, 360, 100), np.ones(100)*66.3, ls='-.', color='light gray')

h = [ax.plot([],[], marker='o', edgecolor='k', facecolor='w', markersize=3, lw=0),
     ax.plot([],[], marker='s', edgecolor='r', facecolor='w', lw=0, ms=5),
     ax.plot([],[], marker='s', edgecolor='r', facecolor='w', lw=0, ms=15)]
ax.legend(h, ['100 km Scene Center', '100 km Boundary', '500 km Boundary'], ncols=1, alpha=1, loc='lr')

#### Row 1, Col 2: SIF
ax = axs[0,1]
sif_timeseries = pd.read_csv(dataloc + '/data/metadata/daily_sea_ice_fraction.csv', index_col=0, parse_dates=True)
sif_timeseries['doy'] = sif_timeseries.index.dayofyear

p90 = sif_timeseries.groupby('doy').quantile(0.9)
for region in p90:
    print(region, p90[region][p90[region] > 0.05].index.max())

linestyles = {region: ls for region, ls in zip(regions.index,
                        ['-', '-.', '--', '-', '-.', '--', '-.', '-', '--'])}
h = []
for region in regions.index:
    h.append(ax.plot(sif_timeseries[region].groupby(sif_timeseries[region].index.dayofyear).median(),
            shadedata=[sif_timeseries[region].groupby(sif_timeseries[region].index.dayofyear).quantile(0.25),
                       sif_timeseries[region].groupby(sif_timeseries[region].index.dayofyear).quantile(0.75)],
            fadedata=[sif_timeseries[region].groupby(sif_timeseries[region].index.dayofyear).quantile(0.1),
                       sif_timeseries[region].groupby(sif_timeseries[region].index.dayofyear).quantile(0.9)],
            c=colors[region], lw=2, ls=linestyles[region], 
            label='')
            )
axs[0,0].legend(h, 
           ['({n}) {t}'.format(n=int(regions.loc[region, 'idx_num']), t=regions.loc[region, 'print_title']) for region in regions.index],
            loc='b', ncols=3,lw=2, order='F')
ax.format(ylabel='Sea Ice Fraction', xlabel='Day of Year', ylim=(0, 1),
         ylocator=np.arange(0.1, 0.91, 0.2), xlim=(60, 274), title='Sea Ice Fraction Climatology')
fig.format(abc=True)

h = [
    ax.plot([],[],lw=2, color='k'), 
    ax.plot([],[],lw=5, alpha=0.5, color='k'), 
    ax.plot([],[],lw=5, alpha=0.25, color='k'), 
]
ax.legend(h, ['Median', '25-75 Pct', '1-99 Pct'], loc='ur', ncols=1)


### Row 2, Col 2: Bar chart for image contents

ax = axs[1, 1]
df = pd.read_csv(dataloc + 'data/validation_dataset/validation_dataset.csv', parse_dates=['start_date'])
df['month'] = df['start_date'].dt.month
fsd_data = {}
for satellite in ['aqua', 'terra']:
    fsd_data[satellite] = {}
    for file in os.listdir(dataloc + '/data/validation_dataset/property_tables/' + satellite):
        if 'csv' in file:
            df_fsd = pd.read_csv(dataloc + '/data/validation_dataset/property_tables/' + satellite + '/' + file)
            df_fsd = df_fsd.loc[~df_fsd.boundary, :].copy()
            
            if len(df_fsd) > 0:
                df_fsd['region'] = file.split('-')[1]
                df_fsd['date'] = pd.to_datetime(file.split('-')[2])
                fsd_data[satellite][file.split('-')[0]] = df_fsd


df_aqua_fsd = pd.concat(fsd_data['aqua']).reset_index()
df_terra_fsd = pd.concat(fsd_data['terra']).reset_index()
df_aqua_fsd.rename({'level_0': 'case_number'}, axis=1, inplace=True)
df_terra_fsd.rename({'level_0': 'case_number'}, axis=1, inplace=True)
df_aqua_fsd.head()

# convert to boolean
for col in ['visible_sea_ice', 'visible_landfast_ice', 'visible_floes', 'visible_water']:
    df[col] = df[col] == 'yes'
df['dummy'] = df.region.notnull()
# df['None'] = [~np.any(x[['visible_sea_ice', 'visible_landfast_ice', 'visible_floes']].notnull()) for idx, x in df.iterrows()]
df.rename({'dummy': 'Total number',
                  'visible_sea_ice': 'Visible ice',
                   'visible_water': 'Visible water',
                  'visible_floes': 'Visible floes',
                  'visible_landfast_ice': 'Visible landfast',
                  }, axis=1, inplace=True)
df2 = df.copy()

ax.bar(df.loc[:,
    ['month', 'Total number', 'Visible ice', 'Visible water', 'Visible floes', 'Visible landfast']].groupby('month').sum(),
    stack=False, cycle='flatui')

print(df.loc[:,
    ['month', 'Total number']].groupby('month').count())
    
ax.legend(loc='ur', ncols=1, fontsize=8)
ax.format(ylabel='Count', xlabel='', xlocator=np.arange(3, 10), title='Summary of Metadata',
          xformatter=['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'], ylim=(0, 75),
          xtickminor=False)



### Stacked: one category per image
# options?
# Low cloud cover, visible floes
# High cloud cover, no visible floes
# Ambiguous
ax = axs[2,1]

df = pd.read_csv(dataloc + 'data/validation_dataset/validation_dataset.csv', parse_dates=['start_date'])
df['month'] = df['start_date'].dt.month
df['region_number'] = [regions.loc[region, 'idx_num'] for region in df.region]
df['Clear Sky, Vis. Floes'] = (df['visible_floes'] == 'yes') & (df['cloud_fraction_manual'] <= 0.2)
df['Cloudy, Vis. Floes'] = (df['visible_floes'] == 'yes') & (df['cloud_fraction_manual'] > 0.2)
df['Clear Sky, No Floes'] = (df['visible_floes'] == 'no') & (df['cloud_fraction_manual'] <= 0.2)
df['Cloudy, No Floes'] = (df['visible_floes'] == 'no') & (df['cloud_fraction_manual'] > 0.2)

ax.bar(df.loc[:,
    ['region_number',
     'Clear Sky, Vis. Floes', 'Cloudy, Vis. Floes', 
     'Clear Sky, No Floes', 'Cloudy, No Floes']].groupby('region_number').sum(),
    stack=True, cycle='flatui_r', width=1/2
)
    
ax.legend(loc='ur', ncols=1, fontsize=8)
ax.format(ylabel='Count', xlabel='Region Number', xlocator=np.arange(1, 10), title='Ice Floe Visibility',
          xtickminor=False)


### FSD
ax = axs[0, 2]
for data, label in zip([df_aqua_fsd, df_terra_fsd], ['Aqua', 'Terra']):
    bins = np.logspace(1, np.log(np.sqrt(100)), 25)*0.25
    x, xb = np.histogram(np.sqrt(data['area'])*0.25, bins)
    ax.plot(0.5*(bins[:-1] + bins[1:]), x, marker='.', label=label)
    ax.format(yscale='log', xscale='log', ylabel='Count', xlocator=(1, 5, 10, 20, 40),
              xlim=(2.5, 50), xlabel='Length scale (km)', title='Floe Size Distribution')
ax.legend(loc='ur', ncols=1)

### Number of floes per image
ax = axs[1, 2]
ax.hist(df.number_floes, bins=30)
ax.format(title='Floes Per Image', ylabel='Number of Floes', xlabel='')

ax = axs[2,2]
dates = pd.to_datetime(df.start_date)
df['decimal_month'] = dates.dt.month + (dates.dt.day - 1) / dates.dt.daysinmonth
for region, data in df.groupby('region'):
    color = colors[region]
    ax.scatter(data.decimal_month, data.number_floes, color=color, m='o', ms=10)
ax.format(title='Floes Per Image', xlabel='', ylabel='Number of Floes',
          xlocator=np.arange(3, 10) + 0.5, xformatter=['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'])

fig.save('../figures/fig_01_sample_summary.png', dpi=300)