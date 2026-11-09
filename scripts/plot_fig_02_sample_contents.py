# Need rio to get the geotransform
import rasterio as rio
from rasterio.plot import reshape_as_image
from skimage.io import imread
import ultraplot as uplt
import numpy as np
import pandas as pd
import geopandas as gpd


dataloc_100km = "../../ice_floe_validation_dataset/"
dataloc_500km = "/Users/dwatkin2/Documents/research/data/ice-floe-validation-dataset/500km/"

case = "011-baffin_bay-500km-20110702"
satellite = "aqua"

tc_100km_rio = rio.open(
    dataloc_100km + "data/modis/truecolor/" + case.replace('500km', '100km') + "." + satellite + ".truecolor.250m.tiff"
)
left_100km, bottom_100km, right_100km, top_100km = tc_100km_rio.bounds
tc_100km_img = tc_100km_rio.read()

tc_500km_rio = rio.open(
    dataloc_500km + "modis_500km/truecolor/" + case + "." + satellite + ".truecolor.250m.tiff"
)
fc_500km_rio = rio.open(
    dataloc_500km + "modis_500km/falsecolor/" + case + "." + satellite + ".falsecolor.250m.tiff"
)

left_500km, bottom_500km, right_500km, top_500km = tc_500km_rio.bounds
tc_500km_img = tc_500km_rio.read()
fc_500km_img = fc_500km_rio.read()

df = gpd.read_file(dataloc_500km + 'calibration_500km/' + case + '-' + satellite + '-samples')

asi_500km_img = imread(dataloc_500km + "asi_500km/sea_ice_concentration/" + case + ".asi.sea_ice_concentration.6250m.png")
nsidc_500km_img = imread(dataloc_500km + "nsidc_500km/sea_ice_concentration/" + case + ".nsidc.sea_ice_concentration.25km.png")
classification = imread('../data/classification_results/500km/011-baffin_bay-500km-20110702.ift.classified.png')
clouds = imread(dataloc_500km + "modis_500km/cloudfraction/" + case + "." + satellite + ".cloudfraction.250m.tiff")
land = imread(dataloc_500km + "modis_500km/landmask/" + case + "." + satellite + ".landmask.250m.tiff")
binary_floes = imread(
    dataloc_100km + "data/validation_dataset/binary_floes/" + case.replace('-500km', '') + "-aqua-binary_floes.png"
)
masie = imread(
    dataloc_100km + "data/masie/sea_ice_extent/" + case.replace('500km', '100km') + ".masie.seaice.250m.tiff"
)

# modis cloud masks
red = clouds[:, :, 0].astype(float)
green = clouds[:, :, 1].astype(float)
blue = clouds[:, :, 2].astype(float)
cloud_free = (green < 1) & (np.abs(red - 102) < 1) & (np.abs(blue - 119) < 1)
cloud_covered = (green < 6) & (red > 250) &  np.abs(blue < 30)

cloud_simple = np.ones(cloud_free.shape)
cloud_simple[cloud_free] = 0
cloud_simple[cloud_covered] = 2

left, bottom = tc_500km_rio.index(left_100km, bottom_100km)
right, top = tc_500km_rio.index(right_100km, top_100km)

fig, axs = uplt.subplots(nrows=2, ncols=4, share=False)
axs[0,0].imshow(reshape_as_image(tc_100km_img))
axs[0,1].imshow(reshape_as_image(tc_500km_img))
axs[1,3].imshow(reshape_as_image(tc_500km_img))
axs[0,2].imshow(reshape_as_image(fc_500km_img))
c = axs[0,3].imshow(cloud_simple/2, cmap='bwr', alpha=0.5, levels=3)
# axs[0,3].colorbar(c, shrink=0.75, ticks=[0.3, 0.6, 0.9])
axs[0, 3].imshow(np.ma.masked_array(land, mask=~land), color='k')
axs[1, 3].imshow(np.ma.masked_array(land, mask=~land), color='k')

axs[1,0].imshow(np.ma.masked_array(masie, mask=masie>0), c='steelblue')
axs[1,0].imshow(np.ma.masked_array(binary_floes, mask=binary_floes==0), color='red5')


c = axs[1,1].imshow(nsidc_500km_img / 255 * 100, cmap='ice', vmin=0, vmax=100)
axs[1,2].imshow(asi_500km_img / 255 * 100, cmap='ice', vmin=0, vmax=100)
# axs[1,3].imshow(classification)

# Draw boxes showing 100 km image bounds
# The images are reshaped, so "bottom" and "top" are "left" and "right" here.
for i in [0, 1]:
    for j in [1, 2, 3]:
        axs[i, j].plot([bottom, top, top, bottom, bottom], [left, left, right, right, left],  color='r', zorder=4)

colors = [c["color"] for c in uplt.Cycle("Dark2", 8)]
category_colors = {
    'none': colors[7],
    'cloudy_ice': colors[2],
    'cloudy_water': colors[0],
    'clear_ice': colors[5],
    'clear_water': colors[6]
}

category_labels = {
    'clear_water': 'Clear-Sky Water',
    'cloudy_water': 'Cloud-Covered Water',
    'clear_ice': 'Clear-Sky Ice',
    'cloudy_ice': 'Cloud-Covered Ice',
    'none': 'NA'
}

category_markers = {
    'clear_water': 'o',
    'cloudy_water': 'o',
    'clear_ice': 'o',
    'cloudy_ice': 'o',
    'none': '^'
}


for name, data in df.groupby('init_cat'):
    m='o'
    edgecolor = 'k'
    facecolor = category_colors[name]
    zorder=3
    if name == 'none':
        # edgecolor = 'w' #
        # facecolor=category_colors[name]
        m='^'
        zorder=2
    axs[1, 3].scatter(
        data.col_center, data.row_center, edgecolor=edgecolor,
        marker=m, facecolor=facecolor, label='', zorder=zorder
        
    )

## Add panel titles
for ax, title in zip(axs,
    ['True Color Image (100 km)', 'True Color Image (500 km)', 'False Color Image (500 km)', 'MODIS Cloud',
     'Manual Floe Labels', 'NSIDC Sea Ice Concentration', 'ASI Sea Ice Concentration', 'Labeled Samples']):
    ax.format(title=title)

## Adjust tick locations
xticks = np.arange(100, 401, 100)
xformatter = [str(int((x - 200)/4)) for x in xticks]
for i in [0,1]:
    axs[i, 0].format(xticks=xticks, yticks=xticks, xformatter=xformatter, yformatter=xformatter, xlabel='Distance (km)', ylabel='Distance (km)')

xticks = np.arange(500, 2001, 500)
xformatter = [str(int((x - 1000)/4)) for x in xticks]
for i in [0, 1]:
    for j in [1, 2, 3]:
        axs[i, j].format(xticks=xticks, yticks=xticks, xformatter=xformatter, yformatter=xformatter, xlabel='Distance (km)', ylabel='Distance (km)')

## Add suptitle, abc
fig.format(abc=True)


## Legends
# Manual labels
axs[1, 0].legend(
    [ax.plot([],[],color='red6', lw=0, m='s'),
     ax.plot([],[],color='steelblue', lw=0, m='s'),
     ax.plot([],[],color='w', edgecolor='k', lw=0, m='s')],
    ['Manual Floe Labels', 'MASIE Water', 'MASIE Ice'],
    ncols=1,
    alpha=1,
    loc='b'
)

axs[0, 3].legend(
    [ax.plot([],[],color='r', alpha=0.5, lw=0, m='s'),
     ax.plot([],[],color='b', alpha=0.5, lw=0, m='s')],
    ['CF > 95%', 'CF < 5%'],
    ncols=1,
    alpha=1,
    loc='lr'
)

axs[1,1].colorbar(c, label='Sea Ice Concentration (%)', loc='b')
axs[1,2].colorbar(c, label='Sea Ice Concentration (%)', loc='b')


axs[1, 3].legend(
    [ax.plot([],[], color=category_colors[c], lw=0, m=category_markers[c], edgecolor='k') 
     for c in category_labels],
    [category_labels[c] for c in category_labels],
alpha=1, ncols=2, loc='b')

fig.save('../figures/fig_02_500km_sample_contents.png', dpi=300)