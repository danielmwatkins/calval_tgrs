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
# First need to run the julia script rotation_test_floe_shapes_ADR.jl
data = []
for fname in os.listdir('../data/rotation_test/'):
    if '.csv' in fname:
        df = pd.read_csv('../data/rotation_test/' + fname)
        df['case'] = fname.split('-')[0].replace('.csv', '')
        if len(df) > 0:
            data.append(df)
df_all = pd.concat(data).reset_index(drop=True)
df_all['floe_id'] = [cn + '_' + str(f).zfill(4) for cn, f in zip(
                                df_all['case'], df_all['floe_id'])]
df_all = df_all.loc[df_all.area > 50]
df_all["L"] = np.sqrt(df_all.area)
df_all['normalized_shape_difference'] = df_all['minimum_shape_difference'] / df_all['area']
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
df_rotation['area_bin'] = np.digitize(df_rotation['area'], bins)

# Divide into testing and training datasets
training_idx = df_testtrain.loc[df_testtrain.satellite == 'aqua', ['case_number', 'training']].set_index('case_number')
df_rotation['training'] = False
df_rotation.loc[training_idx.loc[df_rotation['case']].values.squeeze(), 'training'] = True
df_rot = df_rotation.loc[df_rotation.training, :].copy()

rotated_bin_count = df_rot[['area_bin', 'area']].groupby('area_bin').agg(['mean', 'count'])
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

df_matched['area_bin'] = np.digitize(df_matched['area'], bins)
df_matched = df_matched.merge(
    df_aqua_props, left_on=['case', 'aqua_label'], right_on=['case', 'aqua_label']).merge(
    df_terra_props, left_on=['case', 'terra_label'], right_on=['case', 'terra_label'])

# Divide into testing and training datasets
training_idx = df_testtrain.loc[df_testtrain.satellite == 'aqua', ['case_number', 'training']].set_index('case_number')
df_matched['training'] = training_idx.loc[df_matched['case']].values

df_mg = df_matched.loc[df_matched.training, :].copy()

matched_bin_count = df_matched[['area_bin', 'area']].groupby('area_bin').agg(['mean', 'count'])
matched_bin_count.columns = ['_'.join(col).strip() for col in matched_bin_count.columns.values]


#### Plot ####
fig, axs = pplt.subplots(width=10, height=6, nrows=2, ncols=3, share=False)
ax = axs[0, 0]
h = []
for var, color, offset in zip(['max_adr_area', 'max_adr_convex_area',
                       'max_adr_major_axis_length', 'max_adr_minor_axis_length'],
                      ['tab:blue', 'tab:green', 'tab:orange', 'tab:gray'],
                             np.linspace(-0.3, 0.3, 4)):
    plot_data = df_rot.pivot_table(columns='area_bin', values=var, index=df_rot.index)
    plot_data = plot_data.loc[:, rotated_bin_count['area_count'] > 10]
    x = plot_data.columns.astype(int)
    plot_data.columns = plot_data.columns + offset
    ax.box(plot_data, fillcolor=color,  showfliers=True, marker='.', markersize=1, widths=0.2)
    h.append(ax.plot([],[], m='s', c=color, lw=0, ec='k'))

    # Draw grid lines
    for xc in x[:-1]:
        ax.axvline(xc + 0.5, lw=1, color='gray')
        
ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=x, xlim=plot_range, ylim=(0, 0.3),
          xformatter=[str(int(x)) for x in rotated_bin_count.area_mean.round().values.squeeze()], xrotation=0,
         title='Max.-Min. Abs. Diff. Ratio', xlabel='Bin-average floe area (pixels)', ylabel='Absolute Difference Ratio',
         xgrid=False)

ax.legend(h, ['Area', 'Convex Area', 'Major Axis Length', 'Minor Axis Length'], ncols=1)

ax = axs[1, 0]
h = []
for var, color, offset in zip(['adr_area', 'adr_convex_area', 'adr_major_axis_length', 'adr_minor_axis_length'],
                      ['tab:blue', 'tab:green', 'tab:orange', 'tab:gray'],
                             np.linspace(-0.3, 0.3, 4)):

    idx = df_mg[['terra_cloud_fraction', 'aqua_cloud_fraction']].max(axis=1) < 0.2
    
    plot_data = df_mg.loc[idx, :].pivot_table(columns='area_bin', values=var, index=df_mg.loc[idx, :].index)
    plot_data = plot_data.loc[:, matched_bin_count['area_count'] > 10]
    x = plot_data.columns.astype(int)
    plot_data.columns = plot_data.columns + offset
    ax.box(plot_data, fillcolor=color,  showfliers=True, marker='.', markersize=1,  widths=0.2)
    h.append(ax.plot([],[], m='s', c=color, lw=0, ec='k'))
    for xc in x[:-1]:
        ax.axvline(xc + 0.5, lw=1, color='gray')
        
ax.format(xtickminor=False, #xlocator=bin_area_ave.round().values.squeeze(),
          xlocator=x, xlim=plot_range, ylim=(0, 0.3),
          xformatter=[str(int(x)) for x in matched_bin_count.area_mean.round(1).values.squeeze()], xrotation=0,
         title='Matched Pair Abs. Diff. Ratio', xlabel='Bin-average floe area (pixels)', ylabel='Absolute Difference Ratio',
         xgrid=False)

ax.legend(h, ['Area', 'Convex Area', 'Major Axis Length', 'Minor Axis Length'], ncols=1)


for ax, var, color, dataframe, bin_counts in zip(
        [axs[0,1], axs[0,2], axs[1,1], axs[1,2]],
        ['min_psi_s_correlation', 'max_normalized_shape_difference', 'psi_s_correlation', 'normalized_shape_difference'],
        ['light gray', 'light gray', 'light gray', 'light gray'],
        [df_rotation.loc[df_rotation.training, :], df_rotation.loc[df_rotation.training, :],
         df_matched.loc[df_matched.training, :], df_matched.loc[df_matched.training, :]],
        [rotated_bin_count, rotated_bin_count, matched_bin_count, matched_bin_count]):

    plot_data = dataframe.pivot_table(columns='area_bin', values=var, index=dataframe.index)
    plot_data = plot_data.loc[:, bin_counts['area_count'] > 10]
    x = plot_data.columns.astype(int)
    ax.box(plot_data, fillcolor=color, showfliers=True, marker='.', markersize=1, widths=0.2)
    h.append(ax.plot([],[], m='s', c=color, lw=0, ec='k'))

    # Draw grid lines
    for xc in x[:-1]:
        ax.axvline(xc + 0.5, lw=1, color='gray')
        
    ax.format(xtickminor=False, 
              xlocator=x, xlim=plot_range,
              xformatter=[str(int(x)) for x in bin_counts.area_mean.round().values.squeeze()], xrotation=0,
              xlabel='Bin-average floe area (pixels)',
              xgrid=False, abc=True)
axs[0,1].format(ylabel=r'$\psi-s$ Correlation', title='Max.-Min. Correlation', ylim=(0.69, 1.01))
axs[1,1].format(ylabel=r'$\psi-s$ Correlation', title='Matched Pair Correlation', ylim=(0.69, 1.01))
axs[0,2].format(ylabel='Normalized Shape Difference', title='Max.-Min. Shape Difference', ylim=(-0.02, 0.52))
axs[1,2].format(ylabel='Normalized Shape Difference', title='Matched Pair Shape Difference', ylim=(-0.02, 0.52))

fig.format(abc=True)
fig.save('../figures/fig_XX_ADR_psi-s_normSD.png', dpi=300)