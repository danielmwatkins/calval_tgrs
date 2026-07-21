import numpy as np
import os
import pandas as pd
import ultraplot as pplt
import skimage.io as io
from skimage.morphology import dilation, erosion

# Load the list of cloud clearing evaluation cases
dataloc = '../../ice_floe_validation_dataset/'
df = pd.read_csv(dataloc + '/data/validation_dataset/validation_dataset.csv')
df['case_number'] = [str(cn).zfill(3) for cn in df['case_number']]
df.groupby('region').count()
df['start_date'] = pd.to_datetime(df['start_date'].values)
df.index = [cn + '_' + sat for cn, sat in zip(df.case_number, df.satellite)]

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

# Load raster data and masks
tc_dataloc = dataloc + 'data/modis/truecolor/'
lm_dataloc = dataloc + 'data/validation_dataset/binary_landmask/'
lb_dataloc = dataloc + 'data/validation_dataset/binary_floes/'
lf_dataloc = dataloc + 'data/validation_dataset/binary_landfast/'
ice_mask_loc = '../data/ift_prelim_ice_mask/'
cloud_mask_loc = '../data/ift_cloud_mask/dmw2026/'
modis_mask_loc = '../data/cloudfraction_numeric/'

tc_images = {}
lb_images = {}
lf_images = {}
lm_images = {}
ice_mask_images = {}
cloud_mask_images = {}
modis_mask_images = {}

missing = []
plot_cases = ['011', '022', '104', '108']
for row, data in df.iterrows():
    cn = data['case_number']
    date = pd.to_datetime(data['start_date']).strftime('%Y%m%d')
    region = data['region']
    sat = data['satellite']
    
    if cn in plot_cases:

        prefix = '-'.join([cn, region, '100km', date])        
        tc_images[row] = io.imread(tc_dataloc + '.'.join([prefix, sat, 'truecolor', '250m', 'tiff']))
        lb_images[row] = io.imread(lb_dataloc +  '-'.join([cn, region, date, sat, 'binary_floes' + '.png']))
        lf_images[row] = io.imread(lf_dataloc +  '-'.join([cn, region, date, sat, 'binary_landfast' + '.png']))
        lm_images[row] = io.imread(lm_dataloc +  '-'.join([cn, region, date, sat, 'binary_landmask' + '.png']))
        ice_mask_images[row] = io.imread(ice_mask_loc +  '-'.join([cn, region, '100km', date, sat, '250m', 'prelim_icemask' + '.png']))
        cloud_mask_images[row] = io.imread(cloud_mask_loc +  '-'.join([cn, region, '100km', date, sat, '250m', 'cloudmask' + '.png']))
        modis_mask_images[row] = pd.read_csv(modis_mask_loc + '-'.join([cn, region, date, sat, 'cloudfraction.csv']), index_col=0)
        modis_mask_images[row].index = modis_mask_images[row].index.astype(int)
        modis_mask_images[row].columns = modis_mask_images[row].columns.astype(int)

pplt.rc['abc.bbox'] = True
pplt.rc['abc.bboxalpha'] = 1
pplt.rc['abc.loc'] = 'ul'

fig, ax = pplt.subplots(nrows=4, ncols=4, sharex=False, sharey=True)
titles = []
for i, case_number in enumerate(plot_cases):
    case = case_number + '_aqua'
    region = df.loc[case, 'region'].replace('_', ' ').title()
    titles.append(str(case_number) + ' ' + region)
    ax[0,i].imshow(tc_images[case])

    
    ice_mask = ice_mask_images[case][:,:]
    ax[2,i].imshow(np.ma.masked_array(ice_mask, mask=ice_mask > 0), c=(46/255, 124/255, 163/255))
    ax[2,i].imshow(np.ma.masked_array(ice_mask, mask=ice_mask == 0), c='w')

    cloud_mask = cloud_mask_images[case][:,:]
    ax[2,i].imshow(np.ma.masked_array(cloud_mask, mask=cloud_mask == 0), c=(216/255, 182/255, 240/255))
    
    land_mask = lm_images[case][:,:]
    ax[2,i].imshow(np.ma.masked_array(land_mask, mask=land_mask == 0), c='gray9')
    
    # labeled floes
    if case in lb_images:
        if len(lb_images[case].shape) == 3:
            manual_ice = lb_images[case][:,:,0]
        else:
            manual_ice = lb_images[case][:,:]
        outlines = dilation(manual_ice) - erosion(manual_ice)
        ax[2,i].imshow(np.ma.masked_array(outlines, mask=outlines==0), c='red5', alpha=1)
        
    # labeled landfast
    if case in lf_images:
        if len(lf_images[case].shape) == 3:
            manual_landfast = lf_images[case][:,:,0]
        else:
            manual_landfast = lf_images[case][:,:]
        
        land_buffer = dilation(land_mask)
        manual_landfast[land_buffer > 0] = 255
        outlines = dilation(manual_landfast) - erosion(manual_landfast)
        ax[2,i].imshow(np.ma.masked_array(outlines, mask=outlines == 0), c='yellow4')

    red_band = tc_images[case][:,:,0].copy()
    red_band[cloud_mask > 0] = 0
    red_band[land_mask > 0] = 0
    
    data = np.ravel(red_band)
    data = data[data > 0]
    y, xe = np.histogram(data, bins=np.linspace(0, 255, 64))
    xc = (xe[1:] + xe[:-1]) / 2

    ax[1, i].plot(xc/255, y / np.sum(y), color='red5')
    
    idxmax = y[xc > 75].argmax()
    peak_loc = xc[xc > 75][idxmax]
    ax[1, i].axvline(75/255, lw=1, color='k', ls='--')
    ax[1, i].axvline(peak_loc/255, lw=1, color='k', ls='-.')
    ax[1, i].axvline(0.5 * (75 + peak_loc)/255, lw=1, color='k', ls='-')
    
    ax[1, i].format(ylabel='', xlabel='Band 1 Reflectance')

    # MODIS cloud fraction
    c = ax[3,i].pcolormesh(modis_mask_images[case].values, vmin=0, vmax=100, N=17, cmap='Blues_r')


for j in [0, 2, 3]:
    for i in range(0, 4):
        ax[j, i].format(xticks='none', yticks='none')

h = []
for color in ['w', (46/255, 124/255, 163/255), (216/255, 182/255, 240/255), 'red5', 'yellow4', 'gray8']:
    h.append(ax.plot([],[],m='s', lw=0, c=color, edgecolor='k'))
ax[2,-1].legend(h, ['IFT Sea Ice', 'IFT Water', 'IFT Cloud', 'Manual Floes', 'Manual Landfast Ice', 'Land'], loc='r', ncols=1)

h = []
for ls in ['--', '-.', '-']:
    h.append(ax.plot([],[],m='', lw=1, color='k'))
ax[1,-1].legend(h, ['Minimum Ice', 'Brightness Peak', 'Ice Threshold'], loc='r', ncols=1)

ax[3, -1].colorbar(c, label='MODIS Cloud Fraction (%)')

ax[3,:].format(yreverse=True)

fig.format(abc=True)
fig.format(toplabels=titles, leftlabels=['True Color (1-4-3)', 'Reflectance PDF', 'IFT Masks', 'MODIS Cloud Fraction'])
fig.save("../figures/fig_05_ice_water_discrimination.png", dpi=300)