import pandas as pd
import os
import ultraplot as pplt
import numpy as np

#### Load Data ####
# Define the area bin edges
# bins = np.logspace(1, 4, base=10, num=20)
# Using length scale bin instead
bins = np.arange(0, 100, 10)
bin_centers = 0.5 * (bins[1:] + bins[:-1])
plot_range = (5.5, 13.5)

# Load floe property tables. First need to run get_floe_property_tables.jl.
df_aqua_props = []
for file in os.listdir('../data/floe_property_tables/aqua/'):
    if 'csv' in file:
        df_temp = pd.read_csv('../data/floe_property_tables/aqua/' + file).loc[:, ['label', 'cloud_fraction']]
        df_temp['case'] = file.split('-')[0]
        df_aqua_props.append(df_temp)
df_aqua_props = pd.concat(df_aqua_props)
df_aqua_props['label'] = df_aqua_props['label'].astype(int)
df_aqua_props.rename({'label': 'aqua_label', 'cloud_fraction': 'aqua_cloud_fraction'}, axis=1, inplace=True)

df_terra_props = []
for file in os.listdir('../data/floe_property_tables/terra/'):
    if 'csv' in file:
        df_temp = pd.read_csv('../data/floe_property_tables/terra/' + file).loc[:, ['label', 'cloud_fraction']]
        df_temp['case'] = file.split('-')[0]
        df_terra_props.append(df_temp)
df_terra_props = pd.concat(df_terra_props)
df_terra_props['label'] = df_terra_props['label'].astype(int)
df_terra_props.rename({'label': 'terra_label', 'cloud_fraction': 'terra_cloud_fraction'}, axis=1, inplace=True)

# Get test/train index data
df_testtrain = pd.read_csv('../data/validation_dataset_testtrain_split.csv').rename({'Unnamed: 0': 'case'}, axis=1)
df_testtrain['case_number'] = [str(x).zfill(3) for x in df_testtrain['case_number']]

#### Load the rotation data ######
# First need to run the julia script rotation_test_floe_shapes_ADR.jl
# So we don't oversample the same shapes, select just the Aqua images
data = []
for fname in os.listdir('../data/rotation_test/'):
    if '.csv' in fname:
        if 'aqua' in fname:
            df = pd.read_csv('../data/rotation_test/' + fname)
            df['case'] = fname.split('-')[0].replace('.csv', '')
            if len(df) > 0:
                data.append(df)
df_all = pd.concat(data).reset_index(drop=True)
df_all['floe_id'] = [cn + '_' + str(f).zfill(4) for cn, f in zip(
                                df_all['case'], df_all['floe_id'])]
df_all = df_all.loc[df_all.area > 50]
df_all["L"] = np.sqrt(df_all.area)
# df_all['normalized_shape_difference'] = df_all['minimum_shape_difference'] / df_all['area']
comp_columns = ['area', 'convex_area', 'major_axis_length', 'minor_axis_length',
                'adr_area', 'adr_convex_area', 'adr_major_axis_length',
                'adr_minor_axis_length', 'normalized_shape_difference']

df_init = df_all.loc[df_all.rotation==0, ['floe_id', 'case', 'area', 'perimeter']].set_index('floe_id')
df_max = df_all.groupby('floe_id').max()[comp_columns]
df_max.columns = df_max.add_prefix('max_', axis=1).columns

df_min = df_all[['floe_id', 'psi_s_correlation']].groupby('floe_id').min()
df_min.columns = df_min.add_prefix('min_', axis=1).columns

df_init = df_all.loc[df_all.rotation==0, ['floe_id', 'case', 'area', 'L', 'convex_area', 'major_axis_length',
       'minor_axis_length']].set_index('floe_id')
df_rotation = pd.merge(df_init, df_max, left_index=True, right_index=True).merge(df_min, left_index=True, right_index=True)
df_rotation['L'] = np.sqrt(df_rotation['area'])
# df_rotation['area_bin'] = np.digitize(df_rotation['area'], bins)
df_rotation['length_bin'] = np.digitize(df_rotation['L'], bins)

# Divide into testing and training datasets
training_idx = df_testtrain.loc[df_testtrain.satellite == 'aqua', ['case_number', 'training']].set_index('case_number')
df_rotation['training'] = False
df_rotation.loc[training_idx.loc[df_rotation['case']].values.squeeze(), 'training'] = True
df_rot = df_rotation.loc[df_rotation.training, :].copy()

rotated_bin_count = df_rot[['length_bin', 'L']].groupby('length_bin').agg(['mean', 'count'])
rotated_bin_count.columns = ['_'.join(col).strip() for col in rotated_bin_count.columns.values]

#### Load the matched pairs data ######
# First need to run the julia scripts matched_pairs_test_floe_shapes.jl and get_floe_property_tables.jl
data = []
for fname in os.listdir('../data/matched_pairs_test/'):
    if '.csv' in fname:
        df = pd.read_csv('../data/matched_pairs_test/' + fname)
        df['case'] = fname.split('-')[0].replace('.csv', '')
        if len(df) > 0:
            data.append(df)
df_matched = pd.concat(data).reset_index(drop=True)
df_matched['floe_id'] = [cn + '_' + str(f).zfill(4) for cn, f in zip(
                                df_matched['case'], df_matched['aqua_label'])]
df_matched['area'] = df_matched[['aqua_area', 'terra_area']].mean(axis=1)
df_matched['perimeter'] = df_matched[['aqua_perimeter', 'terra_perimeter']].mean(axis=1)
df_matched['normalized_shape_difference'] = df_matched['minimum_shape_difference'] / df_matched['area']
df_matched = df_matched.loc[df_matched.area > 50]
df_matched["L"] = np.sqrt(df_matched.area)

df_matched['length_bin'] = np.digitize(df_matched['L'], bins)
df_matched = df_matched.merge(
    df_aqua_props, left_on=['case', 'aqua_label'], right_on=['case', 'aqua_label']).merge(
    df_terra_props, left_on=['case', 'terra_label'], right_on=['case', 'terra_label'])

# Divide into testing and training datasets
training_idx = df_testtrain.loc[df_testtrain.satellite == 'aqua', ['case_number', 'training']].set_index('case_number')
df_matched['training'] = training_idx.loc[df_matched['case']].values

df_mg = df_matched.loc[df_matched.training & (df_matched[['terra_cloud_fraction', 'aqua_cloud_fraction']].max(axis=1) < 0.4), :].copy()
   
matched_bin_count = df_matched[['length_bin', 'L']].groupby('length_bin').agg(['mean', 'count'])
matched_bin_count.columns = ['_'.join(col).strip() for col in matched_bin_count.columns.values]



#### Settings #####
n_min = 20
length_min = 10
length_max = 40
xlims = (0.5, 50)
x = np.linspace(xlims[0], xlims[1], 50)

#### Threshold Functions #####
def piecewise_threshold(x, x0, x1, y0, y1):
    if x < x0:
        return y0
    elif x > x1:
        return y1
    else:
        return (y1 - y0)/(x1 - x0)*(x - x1) + y1
        
# Update values below based on the calibration results (99th percentile for above / below value)
adr_area_threshold = lambda x: np.array([piecewise_threshold(y,
                        length_min, length_max,  0.26,  0.06) for y in x])
adr_convex_area_threshold = lambda x: np.array([piecewise_threshold(y,
                        length_min, length_max, 0.25, 0.11) for y in x])
adr_major_axis_threshold = lambda x: np.array([piecewise_threshold(y,
                        length_min, length_max, 0.15, 0.06) for y in x])
adr_minor_axis_threshold = lambda x: np.array([piecewise_threshold(y,
                        length_min, length_max, 0.16, 0.04) for y in x])
scaled_shape_difference_threshold = lambda x: np.array([piecewise_threshold(y,
                        length_min,  length_max, 0.51, 0.24) for y in x])
psi_s_corr_threshold = lambda x: np.array([piecewise_threshold(y,
                        length_min,  length_max, 0.84, 0.97) for y in x])

#### Plotting
fig, axs = pplt.subplots(nrows=2, ncols=3, share=False)

col_means = rotated_bin_count.loc[:, 'L_mean'].round(2)
col_counts = rotated_bin_count.loc[:, 'L_count']
xlims = (0.5, 50)
###### Rotation ########
# 1. ADRs
ax = axs[0, 0]
h = []
for var, color, offset in zip(
    ['max_adr_area', 'max_adr_convex_area', 'max_adr_major_axis_length', 'max_adr_minor_axis_length'],
    ['tab:blue', 'tab:green', 'tab:orange', 'tab:gray'],
    np.linspace(-2.75, 2.75, 4)):
    
    data = df_rot.pivot_table(columns='length_bin', values=var, index=df_rot.index)
    data = data.loc[:, col_counts.values > n_min]
    data.columns = bin_centers[col_counts[col_counts >= n_min].index - 1] + offset
    ax.box(data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=2)
    h.append(ax.plot([],[], m='s', c=color, lw=0, ec='k'))


ax.plot(x, adr_area_threshold(x), color='tab:blue', lw=1, marker='')
ax.plot(x, adr_major_axis_threshold(x), color='tab:orange', lw=1, marker='')
ax.plot(x, adr_convex_area_threshold(x), color='tab:green', lw=1, marker='')
ax.plot(x, adr_minor_axis_threshold(x), color='tab:gray', lw=1, marker='')


for xc in bins:
    ax.axvline(xc + 0.5, lw=1, color='gray')

ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=bin_centers, xlim=xlims, ylim=(0, 0.5),
          xformatter=[str(int(x) * 0.25) for x in bin_centers], xrotation=0,
         title='Max. ADR Under Rotation', xlabel='Floe length scale (km)', ylabel='Absolute Difference Ratio',
         xgrid=True)
ax.legend(h, ['Area', 'Convex Area', 'Major Axis Length', 'Minor Axis Length'], ncols=1)

# 2. Normalized Shape Difference
ax = axs[0,1]
var =  'max_normalized_shape_difference'
color = 'tab:pink'
data = df_rot.pivot_table(columns='length_bin', values=var, index=df_rot.index)
data = data.loc[:, col_counts.values >= n_min]
data.columns = bin_centers[col_counts[col_counts >= n_min].index - 1]
ax.box(data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=2)
ax.plot(x, scaled_shape_difference_threshold(x), color=color, lw=1, marker='')


for xc in bins:
    ax.axvline(xc + 0.5, lw=1, color='gray')

ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=bin_centers, xlim=xlims, ylim=(0, 0.6),
          xformatter=[str(int(x) * 0.25) for x in bin_centers], xrotation=0,
         title='Max.-Min. SD/A Under Rotation', xlabel='Floe length scale (km)', ylabel='Scaled Shape Difference',
         xgrid=True)

# 3. Psi-s Correlation
ax = axs[0, 2]
var = 'min_psi_s_correlation'
color = 'tab:purple'
data = df_rot.pivot_table(columns='length_bin', values=var, index=df_rot.index)
data = data.loc[:, col_counts.values >= n_min]
data.columns = bin_centers[col_counts[col_counts >= n_min].index - 1]
ax.box(data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=2)
ax.plot(x, psi_s_corr_threshold(x), color=color, lw=1, marker='')


for xc in bins:
    ax.axvline(xc + 0.5, lw=1, color='gray')

ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=bin_centers, xlim=xlims, ylim=(0.7, 1.01),
          xformatter=[str(int(x) * 0.25) for x in bin_centers], xrotation=0,
         title='Min. $\\psi$-s Correlation Under Rotation', xlabel='Floe length scale (km)', ylabel='$\\psi$-s Correlation',
         xgrid=True)

##### Matched Pairs ######
col_means = matched_bin_count.loc[:, 'L_mean'].round(2)
col_counts = matched_bin_count.loc[:, 'L_count']
xlims = (0.5, 50)

# 1. ADRs
ax = axs[1, 0]
h = []
for var, color, offset in zip(
    ['adr_area', 'adr_convex_area', 'adr_major_axis_length', 'adr_minor_axis_length'],
    ['tab:blue', 'tab:green', 'tab:orange', 'tab:gray'],
    np.linspace(-2.75, 2.75, 4)):

    data = df_mg.pivot_table(columns='length_bin', values=var, index=df_mg.index)
    data = data.loc[:, [x for x in data.columns if col_counts[x] >= n_min]]
    data.columns = bin_centers[col_counts[col_counts >= n_min].index - 1] + offset
    ax.box(data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=2)
    h.append(ax.plot([],[], m='s', c=color, lw=0, ec='k'))
    
    for xc in bins:
        ax.axvline(xc + 0.5, lw=1, color='gray')

ax.plot(x, adr_area_threshold(x), color='tab:blue', lw=1, marker='')
ax.plot(x, adr_major_axis_threshold(x), color='tab:orange', lw=1, marker='')
ax.plot(x, adr_convex_area_threshold(x), color='tab:green', lw=1, marker='')
ax.plot(x, adr_minor_axis_threshold(x), color='tab:gray', lw=1, marker='')

ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=bin_centers, xlim=xlims, ylim=(-0.01, 0.5),
          xformatter=[str(int(x) * 0.25) for x in bin_centers], xrotation=0,
          title='Matched Pair ADR', xlabel='Floe length scale (km)', ylabel='Absolute Difference Ratio',
          xgrid=False)

ax.legend(h, ['Area', 'Convex Area', 'Major Axis Length', 'Minor Axis Length'], ncols=1)

# 2. Normalized Shape Difference
ax = axs[1,1]
var =  'normalized_shape_difference'
color = 'tab:pink'
data = df_mg.pivot_table(columns='length_bin', values=var, index=df_mg.index)
data = data.loc[:, [x for x in data.columns if col_counts[x] >= n_min]]
data.columns = bin_centers[col_counts[col_counts >= n_min].index - 1]
ax.box(data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=2)
ax.plot(x, scaled_shape_difference_threshold(x), color=color, lw=1, marker='')
for xc in bins:
    ax.axvline(xc + 0.5, lw=1, color='gray')

ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=bin_centers, xlim=xlims, ylim=(0, 0.6),
          xformatter=[str(int(x) * 0.25) for x in bin_centers], xrotation=0,
         title='Matched Pair SD/A', xlabel='Floe length scale (km)', ylabel='Scaled Shape Difference',
         xgrid=True)


# 2. Psi-S Correlation
ax = axs[1,2]
var =  'psi_s_correlation'
color = 'tab:purple'
data = df_mg.pivot_table(columns='length_bin', values=var, index=df_mg.index)
data = data.loc[:, [x for x in data.columns if col_counts[x] >= n_min]]
data.columns = bin_centers[col_counts[col_counts >= n_min].index - 1]
ax.box(data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=2)
ax.plot(x, psi_s_corr_threshold(x), color=color, lw=1, marker='')
for xc in bins:
    ax.axvline(xc + 0.5, lw=1, color='gray')

ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=bin_centers, xlim=xlims, ylim=(0.7, 1.01),
          xformatter=[str(int(x) * 0.25) for x in bin_centers], xrotation=0,
         title='Matched Pair $\\psi$-s Correlation', xlabel='Floe length scale (km)', ylabel='$\\psi$-s Correlation',
         xgrid=True)
fig.format(abc=True)
# TBD: Determine where the small / large cutoffs should be
fig.save('../figures/fig_XX_ADR_psi-s_normSD.png', dpi=300)