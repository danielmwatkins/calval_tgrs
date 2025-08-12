"""Before running, execute the notebook tracker_distance_threshold.ipynb to produce the CSV file with displacement times."""
import pandas as pd
import numpy as np
import ultraplot as pplt
from numpy.polynomial import Polynomial

plot_df = pd.read_csv('../data/iabp_percentile_results.csv', index_col=0)
# coefficients are computed from plot df index (dt in hours) and percentiles converted to km
polyfit_results = Polynomial.fit(np.log10(plot_df.index), np.log10(plot_df['q99']/1e3), deg=2).convert(domain=(-1, 1))

# The polyfit is done in log space.
# So for a given dt in hours, the distance in meters is:
a, b, c = np.round(polyfit_results.coef, 3)
print("Coefficients:", a, b, c)
fitted_line = lambda x: a +  b * x + c * x**2 
dist_thresh = lambda dt: 10**fitted_line(np.log10(dt))

fig, ax = pplt.subplots()
percentiles = ['q01', 'q25', 'median', 'q75', 'q99']
ax.plot(plot_df['median']/1e3, marker='.', shadedata=plot_df[['q25', 'q75']].T/1e3, fadedata=plot_df[['q10', 'q90']].T/1e3, label='')
ax.fill_between(plot_df.index, plot_df['q01']/1e3, plot_df['q99']/1e3, color='tab:blue', alpha=0.1, label='')

t = np.linspace(1, 96)
x = dist_thresh(t)
ax.plot(t, x, color='k', lw=1, ls='--', label='Quadratic Threshold')

h = [ax.plot([],[], color='tab:blue', lw=1, m='.')] + \
    [ax.plot([], [], lw=5, color='tab:blue', alpha=a) for a in [0.5, 0.25, 0.1]] + \
    [ax.plot([],[], color='k', lw=1, ls='--')]

l = ['Median', '25-75%', '10-90%', '1-99%', 'Quadr. Fit']
ax.legend(h, l, ncols=1, loc='lr', alpha=1)

ax.format(ylabel=r'$\Delta X$ (km)', xlabel=r'$\Delta T$ (h)')
ax.format(yscale='log', title='Buoy displacement by elapsed time', xscale='log')
fig.save('../figures/fig_XX_travel_distance.png', dpi=300)