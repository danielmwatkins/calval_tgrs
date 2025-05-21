"""
The approach in this case is based on comparison with the cloud fraction averaged across the whole image, rather than pixel by pixel. It uses the "B or C" version of the algorithm rather than the "B and C" version.
"""
import numpy as np
import os
import pandas as pd
import proplot as pplt
import rasterio as rio
from rasterio.plot import reshape_as_image
from sklearn.model_selection import KFold
from scipy.interpolate import interp1d

# Load the list of cloud clearing evaluation cases
dataloc = '../../ice_floe_validation_dataset/'

df = pd.read_csv(dataloc + 'data/validation_dataset/validation_dataset.csv')
df['case_number'] = [str(cn).zfill(3) for cn in df['case_number']]
df.groupby('region').count()

df['start_date'] = pd.to_datetime(df['start_date'].values)
df.index = [cn + '_' + sat for cn, sat in zip(df.case_number, df.satellite)]

def fname(case_data, imtype='labeled'):
    """Generates filenames from rows in the overview table. imtype can be "labeled", "truecolor", 
    or "falsecolor"."""

    cn = case_data['case_number']
    date = pd.to_datetime(case_data['start_date']).strftime('%Y%m%d')
    region = case_data['region']
    sat = case_data['satellite']
    if imtype=='labeled':
        return  '-'.join([cn, region, date, sat, 'labeled_floes.png'])
        
    elif imtype in ['truecolor', 'falsecolor', 'cloudfraction']:
        prefix = '-'.join([cn, region, '100km', date])
        return '.'.join([prefix, sat, imtype, '250m', 'tiff'])

# Load raster data
fc_dataloc = dataloc + '/data/modis/falsecolor/'
cf_dataloc = dataloc + '/data/modis/cloudfraction_numeric/'

fc_images = {}
cf_images = {}

for row, data in df.iterrows():
    with rio.open(fc_dataloc + fname(df.loc[row,:], 'falsecolor')) as im:
        fc_images[row] = im.read()

#### Manual Estimate #####
# Compute estimated cloud fraction for a range of threshold values
cloud_cases = df.dropna(subset='cloud_fraction_manual').index
TC = np.linspace(0, 200, 21)
df_est_tc = pd.DataFrame(index=cloud_cases,
                          columns=TC, data=np.nan)
# for each value, compute an estimated cloud fraction based on the fc_images
for tc in df_est_tc.columns:
    for case in df_est_tc.index:
        df_est_tc.loc[case, tc] = np.mean(fc_images[case][0,:,:] > tc)

# finally, calculate the difference between F and F_est
df_err = (df_est_tc.T - df.loc[cloud_cases, 'cloud_fraction_manual']).T

# divide into training and test sets
df_training = df_err.sample(frac=2/3, replace=False, random_state=306)
df_testing = df_err.loc[[x for x in df_err.index if x not in df_training.index]]

x_eval = np.linspace(TC.min(), TC.max(), 201)
manual_results_table = []
manual_rmse_curves = []
# Initialize cross validation
# TBD: learn how to stratify within the K-fold samplng
kf = KFold(n_splits=5, random_state=20240126, shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(df_training)):
    # first get the root mean square error as a function of TC
    rmse = np.sqrt(np.mean( df_training.iloc[train_index,:]**2, axis=0))
    manual_rmse_curves.append(rmse)
    # then find the minimum
    tc_optimal = pd.Series(interp1d(x=rmse.index, y=rmse.values, kind='quadratic')(x_eval), index=x_eval).idxmin()

    test_cases = df_training.iloc[test_index, :].index
    test_results = np.array([
        np.mean(fc_images[case][0,:,:] > tc_optimal) - df.loc[case, 'cloud_fraction_manual']
            for case in test_cases])
    tc_rmse = np.sqrt(np.mean(test_results**2))
    manual_results_table.append([tc_optimal, tc_rmse])

manual_results_table = pd.DataFrame(manual_results_table, columns=['TC', 'RMSE'])

print(manual_results_table)
print(manual_results_table.mean(axis=0).round(2))
print('Error against held-out data')

manual_tc = manual_results_table.mean(axis=0).round(2)['TC']
test_results = np.array([
    np.mean(fc_images[case][0,:,:] > manual_tc) - df.loc[case, 'cloud_fraction_manual']
        for case in df_testing.index])
manual_test_results = pd.Series(test_results, index=df_testing.index)
manual_tc_rmse = np.sqrt(np.mean(test_results**2))
print('Manual CF error:', np.round(manual_tc_rmse, 3))

###### Calculation of error against MODIS cloud product ########
# Loading from file
cf_images = {}
for case in df.index:
    file = fname(df.loc[case]).replace('labeled_floes.png', 'cloudfraction.csv')
    cf_images[case] = pd.read_csv("../data/cloudfraction_numeric/" + file, index_col=0) 
    cf_images[case].index = cf_images[case].index.astype(int)
    cf_images[case].columns = cf_images[case].columns.astype(int)

df['cloud_fraction_modis'] = np.nan
for case in df.index:
    df.loc[case, 'cloud_fraction_modis'] = np.mean(cf_images[case]/100)


regions = pd.read_csv('../../eval_seg/data/metadata/region_definitions.csv', index_col=0)

colors = {region: c['color'] for region, c in zip(
            regions.index,
            pplt.Cycle('dark2', len(regions)))}
markerstyles = {region: ls for region, ls in zip(regions.index,
                        ['o', 's', '^', '+', '*', 'd', '.', 'p', 'x'])}


regions['print_title'] = [c.replace('_', ' ').title().replace('Of', 'of') for c in regions.index]
regions = regions.sort_values('center_lon')

#### Plot comparison between MODIS and manual cloud fraction
fig, ax = pplt.subplots()
for region, group in df.groupby('region'):
    
    ax.scatter(group['cloud_fraction_manual'], group['cloud_fraction_modis'], label=regions.loc[region, 'print_title'],
               m=markerstyles[region], color=colors[region])
ax.legend(loc='b', ncols=2)
ax.plot([0, 1], [0, 1])
ax.format(ylim=(-0.05, 1.05), xlim=(-0.05, 1.05), xlabel='Manual Cloud Fraction', ylabel='MODIS Cloud Fraction')
ax.format(lrtitle='$\\rho=$' + str(np.round(df['cloud_fraction_modis'].corr(df['cloud_fraction_manual']), 2)))
fig.save('../figures/manual_modis_cloud_comparison.png', dpi=300)

###### Calculation ##########

# Compute estimated cloud fraction for a range of threshold values
cloud_cases = df.dropna(subset='cloud_fraction_modis').index
TC = np.linspace(0, 200, 21)
df_est_tc = pd.DataFrame(index=cloud_cases,
                          columns=TC, data=np.nan)
# for each value, compute an estimated cloud fraction based on the fc_images
for tc in df_est_tc.columns:
    for case in df_est_tc.index:
        df_est_tc.loc[case, tc] = np.mean(fc_images[case][0,:,:] > tc)

# finally, calculate the difference between F and F_est
df_err = (df_est_tc.T - df.loc[cloud_cases, 'cloud_fraction_modis']).T
df_training = df_err.sample(frac=2/3, replace=False, random_state=306)
df_testing = df_err.loc[[x for x in df_err.index if x not in df_training.index]]

x_eval = np.linspace(TC.min(), TC.max(), 201)

results = []
modis_rmse_curves = []
# Initialize cross validation
# TBD: learn how to stratify within the K-fold samplng
kf = KFold(n_splits=5, random_state=20240126, shuffle=True)

for i, (train_index, test_index) in enumerate(kf.split(df_training)):
    # first get the root mean square error as a function of TC
    rmse = np.sqrt(np.mean( df_training.iloc[train_index,:]**2, axis=0))
    modis_rmse_curves.append(rmse)
    
    # then find the minimum
    tc_optimal = pd.Series(interp1d(x=rmse.index, y=rmse.values, kind='quadratic')(x_eval), index=x_eval).idxmin()

    test_cases = df_training.iloc[test_index, :].index
    test_results = np.array([
        np.mean(fc_images[case][0,:,:] > tc_optimal) - df.loc[case, 'cloud_fraction_modis']
            for case in test_cases])
    tc_rmse = np.sqrt(np.mean(test_results**2))
    results.append([tc_optimal, tc_rmse])
modis_results_table = pd.DataFrame(results, columns=['TC', 'RMSE'])
print(modis_results_table)
print(modis_results_table.mean(axis=0).round(2))

# Error against the held-out data
tc = modis_results_table.mean(axis=0).round(2)['TC']
test_results = np.array([
    np.mean(fc_images[case][0,:,:] > tc) - df.loc[case, 'cloud_fraction_modis']
        for case in df_testing.index])
modis_test_results = pd.Series(test_results, index=df_testing.index)
tc_rmse = np.sqrt(np.mean(modis_test_results**2))
print('MODIS CF error:', np.round(tc_rmse, 3))


##### Plot Results ######
# Possible to re-order so that RMSE still on left, but bar is behind dots?

fig, axs = pplt.subplots(ncols=2, nrows=2, share=False)

for ax, rmse_results in zip([axs[0,0], axs[0,1]], [manual_rmse_curves, modis_rmse_curves]):
    for rmse in rmse_results:
        ax.plot(rmse, label='', color='gray', lw=1) #label='k=' + str(i))
    ax.plot([],[], color='gray', lw=1, label='5-fold CV results')
    # ax.axvline(110, color='k', label='Default threshold')
    
    ax.format(ylabel='RMSE', xlabel='Cloud Threshold')
    ax.legend(loc='ul', ncols=1, alpha=1)
axs[0,0].axvline(manual_tc, color='k', ls='--', label='Optimum threshold')
axs[0,1].axvline(tc, color='k', ls='--', label='Optimum threshold')
axs[0, 0].format(title='Manual Cloud Fraction')
axs[0, 1].format(title='MODIS Cloud Fraction')

for ax, test_results in zip([axs[1,0], axs[1,1]], [manual_test_results, modis_test_results]):

    plot_df = test_results.groupby(df.loc[test_results.index, 'region']).apply(lambda x: np.sqrt(np.mean(x**2)))
    plot_df.index = [regions.loc[r, 'print_title'] for r in plot_df.index]
    
    ax.plot(plot_df,
        marker='o', lw=0, zorder=2, color='k')

    ax2 = ax.twinx()

    plot_df = test_results.groupby(df.loc[test_results.index, 'region']).count()
    plot_df.index = [regions.loc[r, 'print_title'] for r in plot_df.index]
    
    ax2.bar(
        plot_df, zorder=0, color='gray', alpha=0.35
        )
    ax2.format(ylabel='Count', ytickminor=False)
    
    ax.format(xrotation=45, ylim=(0, 1), xlabel='', ylabel='RMSE', title='Validation against held-out data')
    ax.format(urtitle='n=' + str(len(test_results)))
    ax2.format(ylim=(0, 25))
    h = [ax.plot([],[],marker='', ls='--', color='k', lw=1),
         ax.plot([],[],marker='o', color='gray8', lw=0),
         ax.plot([],[],marker='s', color='gray', alpha=0.75, lw=0)]
    ax2.legend(h, ['RMSE (all)', 'RMSE (region)', 'Count'], loc='ul', ncols=1, alpha=1)

axs[1,0].axhline(manual_tc_rmse, color='k', ls='--', lw=1)
axs[1,1].axhline(tc_rmse, color='k', ls='--', lw=1) 
axs.format(abc=True)
fig.save('../figures/cloud_validation_step1_manual_modis.png')

### Figure with only the MODIS comparison
fig, axs = pplt.subplots(ncols=1, nrows=2, share=False)
ax = axs[0]
for rmse in modis_rmse_curves:
    ax.plot(rmse, label='', color='gray', lw=1) #label='k=' + str(i))
ax.plot([],[], color='gray', lw=1, label='5-fold CV results')
ax.axvline(tc, color='k', ls='--', label='Optimum threshold')
ax.format(ylabel='RMSE', xlabel='Cloud Threshold ($\\tau_c$)', title='Cloud Fraction Error Vs. Threshold')
ax.legend(loc='ul', ncols=1, alpha=1)

ax = axs[1]
plot_df = modis_test_results.groupby(df.loc[modis_test_results.index, 'region']).apply(lambda x: np.sqrt(np.mean(x**2)))
plot_df.index = [regions.loc[r, 'print_title'] for r in plot_df.index]

ax.plot(plot_df,
    marker='o', lw=0, zorder=2, color='k')

ax2 = ax.twinx()

plot_df = modis_test_results.groupby(df.loc[modis_test_results.index, 'region']).count()
plot_df.index = [regions.loc[r, 'print_title'] for r in plot_df.index]

ax2.bar(
    plot_df, zorder=0, color='gray', alpha=0.35
    )
ax2.format(ylabel='Count', ytickminor=False)

ax.format(xrotation=45, ylim=(0, 1), xlabel='', ylabel='RMSE', title='')
ax.format(urtitle='n=' + str(len(modis_test_results)))
ax2.format(ylim=(0, 25))
h = [ax.plot([],[],marker='', ls='--', color='k', lw=1),
     ax.plot([],[],marker='o', color='gray8', lw=0),
     ax.plot([],[],marker='s', color='gray', alpha=0.75, lw=0)]
ax2.legend(h, ['RMSE (all)', 'RMSE (region)', 'Count'], loc='ul', ncols=1, alpha=1)

ax.axhline(tc_rmse, color='k', ls='--', lw=1) 
axs.format(abc=True)
fig.save('../figures/cloud_validation_step1_modis_v1.png')

