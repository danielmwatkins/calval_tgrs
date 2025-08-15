import pandas as pd
import os
import ultraplot as pplt
import numpy as np

# Define the area bin edges
bins = np.logspace(1, 4, base=10, num=20)
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
# First need to run the julia script rotation_test_floe_shapes.jl
data = []
for fname in os.listdir('../data/rotation_test/'):
    if '.csv' in fname:
        df = pd.read_csv('../data/rotation_test/' + fname)
        df['case'] = fname.split('-')[0].replace('.csv', '')
        if len(df) > 0:
            data.append(df)
df_all = pd.concat(data)
df_all['floe_id'] = [cn + '_' + str(f).zfill(4) for cn, f in zip(
                                df_all['case'], df_all['floe_id'])]
df_all = df_all.loc[df_all.area > 50]
df_all["L"] = np.sqrt(df_all.area)

df_max = df_all[['floe_id', 'minimum_shape_difference']].groupby('floe_id').max()
df_min = df_all[['floe_id', 'psi_s_correlation']].groupby('floe_id').min()
df_min.columns = df_min.add_prefix('min_', axis=1).columns
df_init = df_all.loc[df_all.rotation==0, ['floe_id', 'case', 'area', 'perimeter']].set_index('floe_id')
df_rotation = pd.merge(df_init, df_min, left_index=True, right_index=True).merge(df_max, left_index=True, right_index=True)

# df_rotation['length_bin'] = np.digitize(df_rotation['L'], bins)
df_rotation['area_bin'] = np.digitize(df_rotation['area'], bins)

# Divide into testing and training datasets
training_idx = df_testtrain.loc[df_testtrain.satellite == 'aqua', ['case_number', 'training']].set_index('case_number')
df_rotation['training'] = False
df_rotation.loc[training_idx.loc[df_rotation['case']].values.squeeze(), 'training'] = True
df_rot = df_rotation.loc[df_rotation.training, :].copy()


# Normalize by perimeter
df_rotation['normalized_shape_difference'] = df_rotation['minimum_shape_difference'] / df_rotation['perimeter']

# Determine bin counts
rotation_bin_count = df_rot[['area_bin', 'area']].groupby('area_bin').mean()
rotation_bin_count['count'] = df_rot[['area_bin', 'area']].groupby('area_bin').count()['area']

#### Load the matched pairs data ######
# First need to run the julia script matched_pairs_test_floe_shapes.jl
data = []
for fname in os.listdir('../data/matched_pairs_test/'):
    if '.csv' in fname:
        df = pd.read_csv('../data/matched_pairs_test/' + fname)
        df['case'] = fname.split('-')[0].replace('.csv', '')
        if len(df) > 0:
            data.append(df)
df_matched = pd.concat(data)
df_matched['floe_id'] = [cn + '_' + str(f).zfill(4) for cn, f in zip(
                                df_matched['case'], df_matched['aqua_label'])]
df_matched['area'] = df_matched[['aqua_area', 'terra_area']].mean(axis=1)
df_matched['perimeter'] = df_matched[['aqua_perimeter', 'terra_perimeter']].mean(axis=1)
df_matched['normalized_shape_difference'] = df_matched['minimum_shape_difference'] / df_matched['perimeter']
df_matched = df_matched.loc[df_matched.area > 50]
df_matched["L"] = np.sqrt(df_matched.area)

df_matched['area_bin'] = np.digitize(df_matched['area'], bins)

# Divide into testing and training datasets
training_idx = df_testtrain.loc[df_testtrain.satellite == 'aqua', ['case_number', 'training']].set_index('case_number')
df_matched['training'] = training_idx.loc[df_matched['case']].values
df_mg = df_matched.loc[df_matched.training, :]

matched_bin_count = df_mg[['area_bin', 'area']].groupby('area_bin').mean()
matched_bin_count['count'] = df_mg[['area_bin', 'area']].groupby('area_bin').count()['area']











#### Plot ####
fig, axs = pplt.subplots(width=8, height=6, nrows=2, share=False, ncols=2)
h = []

for ax, var, color, dataframe, bin_counts in zip(
        axs,
        ['min_psi_s_correlation', 'normalized_shape_difference', 'psi_s_correlation', 'normalized_shape_difference'],
        ['light gray', 'light gray', 'light gray', 'light gray'],
        [df_rotation.loc[df_rotation.training, :], df_rotation.loc[df_rotation.training, :],
         df_matched.loc[df_matched.training, :], df_matched.loc[df_matched.training, :]],
        [rotation_bin_count, rotation_bin_count, matched_bin_count, matched_bin_count]):

    plot_data = dataframe.pivot_table(columns='area_bin', values=var, index=dataframe.index)
    plot_data = plot_data.loc[:, bin_counts['count'] > 10]
    x = plot_data.columns.astype(int)
    ax.box(plot_data, fillcolor=color, showfliers=True, marker='.', markersize=1, widths=0.2)
    h.append(ax.plot([],[], m='s', c=color, lw=0, ec='k'))

    # Draw grid lines
    for xc in x[:-1]:
        ax.axvline(xc + 0.5, lw=1, color='gray')
        
    ax.format(xtickminor=False, 
              xlocator=x, xlim=plot_range,
              xformatter=[str(int(x)) for x in bin_counts.area.round().values.squeeze()], xrotation=0,
              xlabel='Bin-average floe area (pixels)',
              xgrid=False, abc=True)
axs[0,0].format(ylabel=r'$\psi-s$ Correlation', title='Minimum Correlation Under Rotation', ylim=(0.49, 1.01))
axs[1,0].format(ylabel=r'$\psi-s$ Correlation', title='Correlation Between Matched Pairs', ylim=(0.49, 1.01))
axs[0,1].format(ylabel='Normalized Shape Difference', title='Max. Optimal Shape Difference Under Rotation', ylim=(-0.02, 1.02))
axs[1,1].format(ylabel='Normalized Shape Difference', title='Optimal Shape Difference Between Matched Pairs', ylim=(-0.03, 3.03))
fig.save('../figures/fig_XX_correlation_shape_difference.png', dpi=300)