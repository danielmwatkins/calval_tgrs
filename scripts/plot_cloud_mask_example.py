import numpy as np
import pandas as pd
import ultraplot as pplt
import rasterio as rio
from rasterio.plot import reshape_as_image
import skimage

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

    elif imtype == 'cloudfraction_numeric':
        
        return '-'.join([cn, region, date, sat, 'cloudfraction.csv'])


case = '006_terra'
fc_dataloc = dataloc + 'data/modis/falsecolor/'

with rio.open(fc_dataloc + fname(df.loc[case,:], 'falsecolor')) as im:
    fc_img = im.read()

# with rio.open(lm_dataloc + fname(df.loc[case,:], 'binary_landmask')) as im:
#     landmask = im.read()

initial = skimage.io.imread("../data/ift_cloud_mask/initial/006-baffin_bay-100km-20220530-terra-250m-cloudmask.png")
cleaned = skimage.io.imread("../data/ift_cloud_mask/cleaned/006-baffin_bay-100km-20220530-terra-250m-cloudmask.png")

fig, ax = pplt.subplots(ncols=3)
ax[0].imshow(reshape_as_image(fc_img))
ax[1].imshow(initial, cmap='mono_r')
ax[2].imshow(cleaned, cmap='mono_r')
ax[0].format(title='False Color Image')
ax[1].format(title='Threshold Mask')
ax[2].format(title='Cleaned Mask')
fig.format(xticks='none', yticks='none', abc=True)
fig.save('../figures/fig_06_cloud_mask_example.png', dpi=300)