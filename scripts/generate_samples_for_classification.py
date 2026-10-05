import skimage.io as io
import os
import pandas as pd
import numpy as np
import ultraplot as uplt
import scipy
import skimage
from skimage.measure import regionprops_table
import skimage.morphology as skim
import geopandas as gpd
from shapely.geometry import box
import rasterio as rio

dataloc = "../data/validation_dataset/modis_500km/"
asi_dataloc = "../data/validation_dataset/asi_500km/"
nsidc_dataloc = "../data/validation_dataset/nsidc_500km/"
sample_saveloc = '../data/validation_dataset/calibration_500km/'

files = os.listdir(dataloc + 'cloudfraction')
files = [f for f in files if '.tiff' in f]
files.sort()

case_numbers = np.unique([f.split('-')[0] for f in files])

# oversample and reduce to correct distance, then take first keep_n samples
n = 2000
imsize=2000
keep_n = 500

g = np.random.default_rng(seed=2340)
sampled_points = {}
for case in case_numbers:
    points = g.choice(range(8, imsize-8), (2, n)) 
    keep = np.ones(n)
    jcomp = np.arange(0, n)
    min_dx = 50
    d = pd.DataFrame(scipy.spatial.distance.cdist(points.T, points.T),
                     columns=jcomp, index=jcomp)
    for idx in range(0, n):
        if keep[idx] > 0:
            di = d.loc[idx, jcomp != idx]
            keep[di[di <= min_dx].index] = 0
    sampled_points[case] = points[:, keep > 0][:, 0:keep_n]

for file in files:
    cn, region, dx, suffix = file.split('-')
    date, satellite, imtype, res, ftype = suffix.split('.')
    
    # Load cloud masks
    rgb_img = io.imread(dataloc + 'cloudfraction/' + file).astype(float)
    red = rgb_img[:, :, 0].astype(float)
    green = rgb_img[:, :, 1].astype(float)
    blue = rgb_img[:, :, 2].astype(float)
    cloud_free = (green < 1) & (np.abs(red - 102) < 1) & (np.abs(blue - 119) < 1)
    cloud_covered = (green < 6) & (red > 250) &  np.abs(blue < 30)

    # Load false color image and land mask
    fc_img = io.imread((dataloc + 'falsecolor/' + file).replace('cloudfraction', 'falsecolor'))
    land_modis = io.imread((dataloc + 'landmask/' + file).replace('cloudfraction', 'landmask')) > 0

    # Load sea ice concentration masks. 
    prefix = file.split('.')[0]
    sic_nsidc = io.imread(nsidc_dataloc + 'sea_ice_concentration/' + prefix + '.nsidc.sea_ice_concentration.25km.png').astype(float) / 255
    land_nsidc = io.imread(nsidc_dataloc + 'landmask/' + prefix + '.nsidc.landmask.25km.png').astype(float) / 255

    # If year = 2012, sic_asi may be missing -- set to -1 in that case
    try:
        sic_asi = io.imread(asi_dataloc + 'sea_ice_concentration/' + prefix + '.asi.sea_ice_concentration.6250m.png').astype(float) / 255
        land_asi = io.imread(asi_dataloc + 'landmask/' + prefix + '.asi.landmask.6250m.png').astype(float) / 255
    except:
        sic_asi = -1 * np.ones(sic_nsidc.shape) # Sets it all to -1 so that when we take the mean, we can filter easily
        land_asi = sic_asi.copy()

    # Load samples corresponding to the current case
    sample_points = sampled_points[cn]

    # Expand points to patches
    Z = np.zeros((2000, 2000))
    for idx in range(0, len(sample_points.T)):
        Z[sample_points[0, idx], sample_points[1, idx]] = 1
    sample_patches = skim.dilation(Z, skim.footprint_rectangle((16,16)))
    sample_patches = skimage.morphology.label(sample_patches)
    
    df = pd.DataFrame(
        regionprops_table(
            sample_patches,
            fc_img[:, :, :2], # drop the alpha channel
            properties=["area", "label", "bbox", "centroid", "intensity_mean"])
        ).set_index('label')
    
    df.rename({
        'bbox-0': 'min_row',
        'bbox-1': 'min_col',
        'bbox-2': 'max_row',
        'bbox-3': 'max_col',
        'centroid-0': 'row_center',
        'centroid-1': 'col_center',
        'intensity_mean-0': 'b7_mean',
        'intensity_mean-1': 'b2_mean',
        'intensity_mean-2': 'b1_mean',
    }, inplace=True, axis=1
    )
    
    for mask_img, varname in zip(
        [cloud_free, cloud_covered, land_modis, sic_asi, land_asi, sic_nsidc, land_nsidc],
        ['clear', 'cloud', 'land_modis',
         'sic_asi', 'land_asi',
        'sic_nsidc', 'land_nsidc']
        ):
        props = regionprops_table(
            sample_patches, 
            intensity_image=mask_img,
            properties=["label", "intensity_mean"]
        )
        df[varname] = pd.Series(props['intensity_mean'], index=props['label'])

    land_tol = 0.1
    idx = (df.land_modis < land_tol) & (df.land_nsidc < land_tol) & (df.land_asi < land_tol)
    df = df.loc[idx, :].copy()
    
    cloud_tol = 0.9 # Fraction that needs to be covered by mask to "count"
    low_ice_tol = 0.01 # Maximum SIC to classify as water
    high_ice_tol = 0.85 # Minimum SIC to classify as ice
    
    # Default to using ASI ice fraction
    df.loc[:, "clr_water"] = (df["sic_asi"] < low_ice_tol) & (df["clear"] > cloud_tol)
    df.loc[:, "clr_ice"] = (df["sic_asi"] > high_ice_tol) & (df["clear"] > cloud_tol)
    df.loc[:, "cld_water"] = (df["sic_asi"] < low_ice_tol) & (df["cloud"] > cloud_tol)
    df.loc[:, "cld_ice"] = (df["sic_asi"] > high_ice_tol) & (df["cloud"] > cloud_tol)
    
    # Use NSIDC where the ASI data is missing
    idx = df.sic_asi < 0
    df.loc[idx, "clr_water"] = (df.loc[idx, "sic_nsidc"] < low_ice_tol) & (df.loc[idx, "clear"] > cloud_tol)
    df.loc[idx, "clr_ice"] = (df.loc[idx, "sic_nsidc"] > high_ice_tol) & (df.loc[idx, "clear"] > cloud_tol)
    df.loc[idx, "cld_water"] = (df.loc[idx, "sic_nsidc"] < low_ice_tol) & (df.loc[idx, "cloud"] > cloud_tol)
    df.loc[idx, "cld_ice"] = (df.loc[idx, "sic_nsidc"] > high_ice_tol) & (df.loc[idx, "cloud"] > cloud_tol)

    df['init_cat'] = 'none'
    df.loc[df.clr_water, 'init_cat'] = 'clear_water'
    df.loc[df.clr_ice, 'init_cat'] = 'clear_ice'
    df.loc[df.cld_water, 'init_cat'] = 'cloudy_water'
    df.loc[df.cld_ice, 'init_cat'] = 'cloudy_ice'

    # Copied column -- category for samples after manual check
    df['manual_cat'] = df['init_cat']
    
    ref_img = rio.open((dataloc + 'falsecolor/' + file).replace('cloudfraction', 'falsecolor'))
    
    # Since the image coordinates are reversed, the largest row number will be the lowest y value.
    x_min, y_min = ref_img.xy(row=df['max_row'], col=df['min_col'])
    x_max, y_max = ref_img.xy(row=df['min_row'], col=df['max_col'])
    df['left_x'] = x_min
    df['right_x'] = x_max
    df['top_y'] = y_max
    df['bottom_y'] = y_min
    
    df['geometry'] = [box(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max in 
                  zip(df.left_x, df.bottom_y, df.right_x, df.top_y)]
    folder = file.replace('.cloudfraction.250m.tiff', '').replace('.', '-') + '-samples'
    os.makedirs(sample_saveloc + folder, exist_ok=True)
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=ref_img.crs)
    gdf.to_file(sample_saveloc + folder + '/' + file.replace('cloudfraction.250m.tiff', 'samples.shp'))
        
    del fc_img, rgb_img, sic_nsidc, land_nsidc, sic_asi, land_asi, cloud_free, cloud_covered, df, gdf
    ref_img.close()
    