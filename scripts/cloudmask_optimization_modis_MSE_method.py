import numpy as np
import os
import pandas as pd
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

# Loading from file
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

def get_binned_ift_est(fc, tc_threshold):
    """Computes a 5 km average of the IFT data and bins it into the
    same resolution as the MODIS cloud fraction data."""
    ift_est = fc > tc_threshold
    
    A = ift_est.astype(int)
    n = A.shape[0]
    m = int(n/20)
    
    # reshapes to dimension (80, 20, 80, 20)
    Amean = A.reshape([m, n//m, m, n//m]).mean(3).mean(1)
    
    bin_edges = np.linspace(0, 100, 17)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    
    ift_vec = np.ravel(Amean*100)
    ift_dig = np.digitize(ift_vec, bin_edges) - 1 # Digitize puts 0 if less than the lowest bin, so you have to subtract.
    ift_dig[ift_dig == 16] = 15 # Put values of exactly 100 into the highest bin
    binned_ift = np.reshape(bin_centers[ift_dig], Amean.shape)

    return binned_ift

def get_binned_landmask(lm, lm_threshold=0.5):
    """Bins the land mask value and returns boolean array where True if land is greater than 50%"""
    L = (lm == 0).astype(int)
    n = L.shape[0]
    m = int(n/20)
    
    # reshapes to dimension (80, 20, 80, 20)
    Lmean = L.reshape([m, n//m, m, n//m]).mean(3).mean(1)
    return Lmean > lm_threshold

# Compute estimated cloud fraction for a range of threshold values
cloud_cases = df.index
TC = np.linspace(0, 200, 41)
df_err = pd.DataFrame(index=cloud_cases,
                          columns=TC, data=np.nan)

# for each value, compute an estimated cloud fraction based on the fc_images
for tc in df_err.columns:
    for case in df_err.index:
        if case in fc_images:
            ift_est = get_binned_ift_est(fc_images[case][0,:,:], tc)
            land = get_binned_landmask(lm_images[case][0,:,:], 0.5) # check whether I want this or its inverse
            mod_est = cf_images[case].values[::20,::20]
            difference = np.ma.masked_array(ift_est - mod_est, mask=land < 0.5)                
            df_err.loc[case, tc] = np.sqrt(np.mean((difference**2)))
df_training = df_err.loc[df['training']]
df_testing = df_err.loc[~df['training']] # only actually need the index here

# Test version without splitting by regions to get syntax
fig, axs = pplt.subplots(nrows=2, share=False)

x_eval = np.linspace(TC.min(), TC.max(), 201)
results = []

kf = KFold(n_splits=5, random_state=20240126, shuffle=True)
ax = axs[0]
for i, (train_index, test_index) in enumerate(kf.split(df_training)):
    # first get the root mean square error as a function of TC
    rmse = np.mean(df_training.iloc[train_index,:], axis=0)
    
    # then find the minimum
    tc_optimal = pd.Series(interp1d(x=rmse.index, y=rmse.values, kind='quadratic')(x_eval), index=x_eval).idxmin()

    test_cases = df_training.iloc[test_index, :].index
    test_results = np.array([
        np.mean(fc_images[case][0,:,:] > tc_optimal) - df.loc[case, 'cloud_fraction_manual']
            for case in test_cases if case in fc_images])
    tc_rmse = np.sqrt(np.mean(test_results**2))
    results.append([tc_optimal, tc_rmse])
    
    ax.plot(rmse, label='', color='gray', lw=1) #label='k=' + str(i))


ax.plot([],[], color='gray', lw=1, label='5-fold CV results')
ax.format(ylim=(0, 100), urtitle='n=' + str(len(df_training)))
kfold_results = pd.DataFrame(results, columns=['TC', 'RMSE']).mean(axis=0).round(2)
tc = kfold_results['TC']

ax = axs[1]
test_results = pd.Series(np.nan, index=df_testing.index)
for case in test_results.index:
     ift_est = get_binned_ift_est(fc_images[case][0,:,:], tc)
     land = get_binned_landmask(lm_images[case][0,:,:], 0.5) # check whether I want this or its inverse
     mod_est = cf_images[case].values[::20,::20]
     difference = np.ma.masked_array(ift_est - mod_est, mask=land < 0.5)                
     test_results.loc[case] = np.sqrt(np.mean((difference**2)))

mean_rmse = test_results.groupby(df.loc[test_results.index, 'region']).mean()
stdev_rmse = test_results.groupby(df.loc[test_results.index, 'region']).std() 
ax.errorbar(x=np.arange(len(mean_rmse)),
    y=mean_rmse.values,
    yerr = stdev_rmse.values,
    marker='o', lw=0, zorder=2, color='k', elinewidth=1)
ax.axhline(test_results.mean(), color='k', ls='--', lw=1)
ax2 = ax.twinx()
region_counts = test_results.groupby(df.loc[test_results.index, 'region']).count()
region_counts = region_counts.loc[region_order]
region_counts.index =  [title_case[r] for r in region_counts.index]

ax2.bar(region_counts, zorder=0, color='gray', alpha=0.35
    )
ax2.format(ylabel='Count', ytickminor=False)

ax.format(xrotation=45, ylim=(0, 100), xlabel='', ylabel='RMSE')
ax.format(urtitle='n=' + str(len(test_results)))
ax2.format(ylim=(0, 25))
h = [ax.plot([],[],marker='', ls='--', color='k', lw=1),
     ax.plot([],[],marker='o', color='gray8', lw=0),
     ax.plot([],[],marker='s', color='gray', alpha=0.75, lw=0)]
ax2.legend(h, ['Mean RMSE (all)', 'Mean RMSE (region)', 'Count'], loc='ul', ncols=1, alpha=1)


# ax.axvline(110, color='k', label='Default threshold')
axs[0].axvline(kfold_results['TC'], color='k', ls='--', label='Optimum threshold')
axs[0].format(ylabel='RMSE (%)', xlabel='Cloud Threshold')
axs[1].format(ylabel='RMSE (%)')

axs[0].legend(loc='lr', ncols=1, alpha=1)
fig.format(abc=True)

fig.save('../figures/cloud_fraction_step1_calibration_modis.png', dpi=300)

kfold_results = pd.DataFrame(results, columns=['TC', 'RMSE'])
print('Cross validation results')
print(kfold_results.round(2))
print("\n")

tc, RMSE = kfold_results.mean(axis=0).round(2)
print('Training data')
print("Cloud threshold: ", int(tc))
print("RMSE (%): ", 100*RMSE)
print("\n")
# Error against the held-out data

print('Comparison against test data')
print('RMSE (%):', test_results.mean().round(2))