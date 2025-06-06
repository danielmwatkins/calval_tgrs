"""Cloud fraction data from MODIS snapshots is available as an RGB image. Here, we use a colormap to translate the RGB values.
After extracting the RGB values, we make a lookup table. Then for each pixel in the image, we find the one the minimizes the RMSE 
error between pixel RGB and lookup table RGB."""

import skimage
plot_examples = True
recompute = False

key = skimage.io.imread("../data/color_key.png")
vals = pd.DataFrame(np.mean(key[:, :, :-1], axis=0), columns=['r', 'g', 'b']).round(0).astype(int)
bin_mean_values = vals2.iloc[6::14,:].astype(int)[['r', 'g', 'b']]

n = len(bin_mean_values)
bin_edges = np.linspace(0, 100, n+1)
bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
bin_mean_values.index = bin_centers

cf_images = {}
if recompute:
    for case in cases:
        color_im = reshape_as_image(cl_images[case])
        cloudfrac_im = np.zeros((color_im.shape[0], color_im.shape[1]))
        for i in range(cloudfrac_im.shape[0]):
            for j in range(cloudfrac_im.shape[1]):
                p = color_im[i, j, :-1]
                cloudfrac_im[i, j] = np.sqrt(np.sum((bin_mean_values - p)**2, axis=1)).idxmin()
        cf_images[case] = cloudfrac_im
        print(case)
        
    for case in cf_images:
        file = fname(df.loc[case], 'binary_landmask').replace('landmask.png', 'cloudfraction.csv')
        pd.DataFrame(cf_images[case]).to_csv("../data/cloudfraction_numeric/" + file)         


if plot_examples:
    fig, ax = pplt.subplots()
    ax.imshow(key, extent=[0, 100, 0, 10])
    ax.format(yformatter='none', xlabel='Cloud fraction (%)')
    fig.save('../figures/colorbar_cloudfrac.png', dpi=300)
    
    fig, ax = pplt.subplots(width=10, height=6)
    vals2 = vals.rolling(7, center=True, min_periods=6).mean()
    for c, channel in zip(['r', 'g', 'b'], range(3)):  
        ax.plot(vals[c], marker='.', color=c)
        ax.plot(vals2[c], color=c, ls='--')
    for x in np.arange(6, len(vals), 14):
        ax.axvline(x)
    fig.save('../figures/colorbar_RGB_values.png', dpi=300)