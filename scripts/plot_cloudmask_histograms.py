import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.simplefilter("ignore")
import proplot as pplt
import rasterio as rio
from rasterio.plot import reshape_as_image
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d

# preliminaries
region_order = ['greenland_sea', 'barents_kara_seas', 'laptev_sea', 'sea_of_okhostk',
                'east_siberian_sea', 'bering_chukchi_seas', 'beaufort_sea', 'hudson_bay', 'baffin_bay']

title_case = {'baffin_bay': 'Baffin Bay',
              'barents_kara_seas': 'Barents-Kara Seas',
              'beaufort_sea': 'Beaufort Sea',
              'bering_chukchi_seas': 'Bering-Chukchi Seas',
              'east_siberian_sea': 'East Siberian Sea',
              'greenland_sea': 'Greenland Sea',
              'hudson_bay': 'Hudson Bay',
              'laptev_sea': 'Laptev Sea',
              'sea_of_okhostk': 'Sea of Okhostk'}

# load the list of cloud clearing evaluation cases
dataloc = '../../ice_floe_validation_dataset/'
df = pd.read_csv(dataloc + '/data/validation_dataset/validation_dataset.csv')
df['case_number'] = [str(cn).zfill(3) for cn in df['case_number']]
df.groupby('region').count()
df['start_date'] = pd.to_datetime(df['start_date'].values)
df.index = [cn + '_' + sat for cn, sat in zip(df.case_number, df.satellite)]

# Divide into testing and training datasets
training_idx = df.sample(frac=2/3, random_state=97234).sort_index().index
df['training'] = False
df.loc[training_idx, 'training'] = True

df.to_csv('../data/validation_dataset_testtrain_split.csv')

# defaults
T1 = 190
T2 = 200
R_lower = 0
R_upper = 0.75
tc = 110

params = {'Original': {'tau_0': 110,
                       'tau_7': 200,
                       'tau_2': 190,
                       'tau_r': 0.75,
                       'c': 'k',
                       'ls': '-'},
          'Hybrid': {'tau_0': 53,
                       'tau_7': 130,
                       'tau_2': 169,
                       'tau_r': 0.53,
                        'c': 'k',
                        'ls': '--'}}

bands = ['b1', 'b2', 'b3', 'b4', 'b7']


def cloud_mask(b2, b7, t0=110, t_b7=200, t_b2=190, t_b7b2=0.75):
    """Cloud mask from the band-level data"""
    step_1 = b7 > t0
    step_2 = b7 < b2*t_b7b2
    step_3 = (b7 < t_b7) & (b2 > t_b2)
    return step_1 & ~(step_2 & step_3)

# translated implementation of cloud mask from Julia code
# assuming false_color_image has band as the first axis
def _get_masks(false_color_image, t0, t_b7, t_b2, r_lower, r_upper):

    false_color_image_b7 = false_color_image[0, :, :]
    false_color_image_b2 = false_color_image[1, :, :]
    clouds_view = false_color_image_b7 > t0
    mask_b7 = false_color_image_b7 < t_b7
    mask_b2 = false_color_image_b2 > t_b2

    # First find all the pixels that meet threshold logic in band 7 (channel 1) and band 2 (channel 2)
    # Masking clouds and discriminating cloud-ice
    mask_b7b2 = mask_b7 & mask_b2

    # Next find pixels that meet both thresholds and mask them from band 7 (channel 1) and band 2 (channel 2)
    b7_masked = np.multiply(mask_b7b2, false_color_image_b7)
    b2_masked = np.multiply(mask_b7b2, false_color_image_b2)

    cloud_ice = b7_masked / b2_masked
    mask_cloud_ice = (cloud_ice >= r_lower) & (cloud_ice < r_upper)
    return mask_cloud_ice, clouds_view

    
def create_cloudmask(false_color_image, t0=110, t_b7=200, t_b2=190, r_lower=0, r_upper=0.75):
    
    mask_cloud_ice, clouds_view = _get_masks(false_color_image, t0, t_b7, t_b2, r_lower, r_upper)

    cloudmask = mask_cloud_ice | ~clouds_view
    return ~cloudmask

def fname(case_data, imtype='labeled_floes'):
    """Generates filenames from rows in the overview table. imtype can be 'labeled_floes', 
    'binary_floes', 'binary_landfast', or 'binary_landmask', 'truecolor', or 'falsecolor'.
    The imtype determines whether a 'png' or 'tiff' is returned.
    """

    cn = case_data['case_number']
    date = pd.to_datetime(case_data['start_date']).strftime('%Y%m%d')
    region = case_data['region']
    sat = case_data['satellite']
    if 'binary' in imtype:
        return  '-'.join([cn, region, date, sat, imtype + '.png'])
        
    elif imtype in ['truecolor', 'falsecolor', 'cloudfraction', 'labeled_floes',]:
        prefix = '-'.join([cn, region, '100km', date])
        return '.'.join([prefix, sat, imtype, '250m', 'tiff'])

    elif imtype in ['seaice', 'landmask',]:
        prefix = '-'.join([cn, region, '100km', date])
        return '.'.join([prefix, 'masie', imtype, '250m', 'tiff'])        

def draw_mask(t0=110,  t7=200, t2=190, r_lower=0, r_upper=0.75, variant="LSW2019"):
    """Returns list of x/y pairs to plot mask partition curve. r_lower not used at all for now."""
    if variant=="LSW2019":
        intersect_t2 = (t2, r_upper*t2)
        intersect_t7 = (t7/r_upper, t7)
        intersect_t0 = (t0/r_upper, t0)
        # If t7 is below t0, then no other unmasking can happen
        if t7 <= t0:
            # print('Case 1')
            x = [0, 255]
            y = [t0, t0]
            return x, y

        # If t2*r_upper is greater than t7, then the ratio is not used
        elif t2*r_upper >= t7:
            # print('Case 2')
            x = [0, t2, t2, 255]
            y = [t0, t0, t7, t7]
            return x, y

        # If the intersection of t7 and the ratio line is outside the range
        # of pixel intensities, then the t7 threshold is not used.
        elif t7/r_upper >= 255: 
            
            # If the intersection between the ratio and the t2 threshold
            # is below the t0 threshold, we get a three-point line
            if r_upper*t2 <= t0: 
                # print('Case 3')
                x = [0, t0/r_upper, 255]
                y = [t0, t0, 255*r_upper]
                return x, y

            # Otherwise, we get a four-point curve with vertical line at the t2 threshold
            # This is the one used by default. 
            else:
                # print('Case 4')
                x = [0, t2, t2, 255]
                y = [t0, t0, r_upper*t2, 255*r_upper]
                return x, y
                
        # Finally we look at where the t7 threshold matters
        else:

            # If the t2 threshold and the ratio line intersect below the t0
            # threshold, then t2 threshold doesn't do anything and we get
            # a four-point curve with horizontal line set by t7
            if r_upper*t2 <= t0:
                # print('Case 5')
                x = [0, t0/r_upper, t7/r_upper, 255]
                y = [t0, t0, t7, t7]
                return x, y

            # Otherwise, we get the only case where all the thresholds matter.
            else:
                # print('Case 6')
                # Five point curve with vertical and horizontal lines
                x = [0, t2, t2, t7/r_upper, 255]
                y = [t0, t0, r_upper*t2, t7, t7]
                return x, y
        
                # Other possibility: what if t0 = 0?


    
##### Load raster data and masks
fc_dataloc = dataloc + 'data/modis/falsecolor/'
tc_dataloc = dataloc + 'data/modis/truecolor/'
cl_dataloc = dataloc + 'data/modis/cloudfraction/'

lm_dataloc = dataloc + 'data/validation_dataset/binary_landmask/'
lb_dataloc = dataloc + 'data/validation_dataset/binary_floes/'
lf_dataloc = dataloc + 'data/validation_dataset/binary_landfast/'

masie_ice_loc = dataloc + 'data/masie/seaice/'
masie_land_loc = dataloc + 'data/masie/landmask/'

tc_images = {}
fc_images = {}
cl_images = {}
lb_images = {}
lf_images = {}
lm_images = {}
mi_images = {}
ml_images = {}

missing = []
for row, data in df.iterrows():
    for datadir, imtype, data_dict in zip([tc_dataloc, fc_dataloc, cl_dataloc,
                                           lb_dataloc, lf_dataloc, lm_dataloc,
                                           masie_ice_loc, masie_land_loc],
                                          ['truecolor', 'falsecolor', 'cloudfraction',
                                           'binary_floes', 'binary_landfast', 'binary_landmask',
                                           'seaice', 'landmask'],
                                          [tc_images, fc_images, cl_images,
                                           lb_images, lf_images, lm_images,
                                           mi_images, ml_images]):
        try:
            with rio.open(datadir + fname(df.loc[row,:], imtype)) as im:
                data_dict[row] = im.read()
        except:
            if imtype in ['falsecolor', 'cloudfraction', 'landmask']:
                print('Couldn\'t read', fname(df.loc[row,:], imtype), imtype)
            elif imtype == 'binary_floes':
                if df.loc[row, 'visible_floes'] == 'yes':
                    missing.append(fname(df.loc[row,:], imtype))
            elif imtype == 'binary_landfast':
                if df.loc[row, 'visible_landfast_ice'] == 'yes':
                    missing.append(fname(df.loc[row,:], imtype))
            elif imtype in ['seaice', 'landmask']: # masie images
                missing.append(fname(df.loc[row,:], imtype))




# Load the numeric cloud fraction data
cf_images = {}
cases = [c for c in cl_images]
cases.sort()

# Make the index of the df a unique case label
df.index = [x.case_number + '_' + x.satellite for row, x in df.iterrows()]

##### Loading cloud fraction from file
cf_images = {}
for case in df.index:
    file = fname(df.loc[case], 'binary_landmask').replace('binary_landmask.png', 'cloudfraction.csv')
    try:
        cf_images[case] = pd.read_csv("../data/cloudfraction_numeric/" + file, index_col=0) 
        cf_images[case].index = cf_images[case].index.astype(int)
        cf_images[case].columns = cf_images[case].columns.astype(int)
    except:
        print(case)

df['cloud_fraction_modis'] = np.nan
for case in cf_images:
    df.loc[case, 'cloud_fraction_modis'] = np.mean(cf_images[case]/100)



pixel_data = {'cloudy': {},
              'clearsky_ice': {},
              'cloudy_ice': {},
              'water': {}
             }

# initialize
for s in pixel_data:
    for b in bands:
        pixel_data[s][b] = []
    pixel_data[s]['case'] = []

# minimum pixel cloud cover for us to call it "cloudy"
min_cloud_frac = 20 

for row, rowdata in df.iterrows():
    case = '{cn}_{s}'.format(cn=rowdata.case_number, s=rowdata.satellite)
    data = {'b1': tc_images[case][0,:,:],
            'b3': tc_images[case][2,:,:],
            'b2': fc_images[case][1,:,:],
            'b4': tc_images[case][1,:,:],
            'b7': fc_images[case][0,:,:]
           }
    
    # get boolean masks for cloud and land
    modis_cloud = cf_images[case] > min_cloud_frac
    land = np.sum(lm_images[case], axis=0) > 1 # replace with boolean mask in the end
    
    if case in lb_images:
        if rowdata.floe_obscuration in ['none', 'light']:
            ice_floes = lb_images[case][0,:,:] > 0
            fast_ice = lf_images[case][0,:,:] > 0
            ## Alternatively can use landfast ice to train the algorithm
            ## Note though that landfast ice is often brighter than floes
            # ice_nocloud_pixels = (ice_floes | fast_ice) & ~modis_cloud
            # ice_cloud_pixels = (ice_floes | fast_ice) & modis_cloud                
 
            ice_nocloud_pixels = ice_floes & ~modis_cloud
            ice_cloud_pixels = ice_floes & modis_cloud                
            
            for band in data:
                pixel_data['clearsky_ice'][band].append(data[band][ice_nocloud_pixels])
                pixel_data['cloudy_ice'][band].append(data[band][ice_cloud_pixels])
            pixel_data['clearsky_ice']['case'].append([case] * ice_nocloud_pixels.astype(int).sum().sum())
            pixel_data['cloudy_ice']['case'].append([case] * ice_cloud_pixels.astype(int).sum().sum())

    elif rowdata.visible_sea_ice == 'no':
        if rowdata.visible_landfast_ice == 'no': # could relax this and just not include where the landfast ice is
            if rowdata.cloud_category_manual == 'opaque':
                cloud_pixels = modis_cloud & ~land
                for band in data:
                    pixel_data['cloudy'][band].append(data[band][cloud_pixels])
                pixel_data['cloudy']['case'].append([case] * cloud_pixels.astype(int).sum().sum())

    if np.mean(modis_cloud & ~land) < 0.75:
        # flag potential water pixels using band 1 intensity
        # Makynen et al. use 25.5 as their threshold (R=0.1 = 25.5/255)
        water_no_cloud_pixels = ~land & ~modis_cloud & (data['b1'] < 26)
        for band in data:
            pixel_data['water'][band].append(data[band][water_no_cloud_pixels])
        pixel_data['water']['case'].append([case] * water_no_cloud_pixels.astype(int).sum().sum())

for s in pixel_data:
    for b in bands:
        pixel_data[s][b] = np.hstack(pixel_data[s][b]).astype(float)
    pixel_data[s]['b7b2'] = pixel_data[s]['b7'] / pixel_data[s]['b2']
    pixel_data[s]['case'] = np.hstack(pixel_data[s]['case'])
    pixel_data[s] = pd.DataFrame(pixel_data[s])
    pixel_data[s]['training'] = df.loc[pixel_data[s]['case'], 'training'].values


fig, axs = pplt.subplots(ncols=3, refwidth=2.3, share=False)

for ax, category, color in zip(axs, ['cloudy', 'cloudy_ice', 'clearsky_ice'], ['steelblue', 'tangerine', 'tab:green']):

    idx = pixel_data[category].training
    x = pixel_data[category].loc[idx, 'b2']
    y = pixel_data[category].loc[idx, 'b7']
    x.name = 'Band 2' # weirdly the xlabel function isn't working at all
    y.name = 'Band 7'
    
    bins = np.arange(0, 261, 10)
    binc = 0.5*(bins[1:] + bins[:-1])
    H, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    H = H.T

    # Estimate levels for contours
    Hd = H/np.sum(H)
    F = np.linspace(0, np.max(Hd), 50)
    L = [1-np.sum(Hd[Hd > f]) for f in F]
    results = pd.Series(L, index=F)
    results = results.loc[~results.duplicated()]
    
    pct = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    levels = interp1d(results, results.index)(pct)
    levels = [x for x in levels] + [np.max(Hd)]
    ax.scatter(x.values, y.values, ms=1, m='.', color='light gray')
    cmap = pplt.Colormap((color, 'w'), l=(100, 0), name='noname', space='hpl')
    
    ax.contourf(binc, binc, np.ma.masked_array(Hd, Hd == 0),
              levels=levels, cmap=cmap, alpha=1)
    ax.contour(binc, binc, Hd, vmin=0, vmax=np.max(Hd),
               levels=levels, lw=1, color='k')
  
    ax.format(titleabove=True,
              title=category.replace('_', ' ').title(),
              ylim=(0, 255), xlim=(0, 255),
              xlabel='Band 2')
    py = ax.panel('r', space=0)
    py.histh(y, bins, color=color, fill=True, ec='k')

    px = ax.panel('t', space=0)
    px.hist(x, bins, color=color, fill=True, ec='k')
    
    pctiles = [0.05, 0.5, 0.95]
    lwidth = [1, 3, 1]
    for p, l in zip(pctiles, lwidth):
        py.axhline(y.quantile(p), lw=l, color='k')
        px.axvline(x.quantile(p), lw=l, color='k')
        
    py.format(grid=False, xlocator=[], ylabel='Band 7', xlabel='', xreverse=False)
    px.format(grid=False, ylocator=[])

for ax in axs:
    for method, title, ls in zip(['Original', 'Hybrid'], ['LSW2019', 'Updated'], ['-', '--']):
        x, y = draw_mask(t0=params[method]['tau_0'],
                         t2=params[method]['tau_2'],
                         t7=params[method]['tau_7'],
                         r_upper=params[method]['tau_r'])
        ax.plot(x, y, label=title, ls=ls, lw=2, color='k')

ax.legend(ncols=1)
fig.save('../figures/fig_05_cloud_mask_histograms_update.png', dpi=300)