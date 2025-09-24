"""
Filter tiles based on the image contrast, land mask, and cloud mask.

The filter function is a fitted 4-parameter logit function trained on a set of 18 manually validated 500 km by 500 km images.
(9 cases, 2 satellite images per case). Each image was divided into 25 km tiles (400 tiles per image) then a set of image
properties were computed from each tile. Based on the size of the coefficients in the function and the strength of correlation
between image measures, the parameters chosen are

1. Robust contrast of the Band 1 (red) in the truecolor image (1-99 percentile range)
2. Robust contrast of the Band 7 (shortwave infrared) in the falsecolor image (1-99 percentile range)
3. Fraction of clear-sky pixels with Band 2 (near infrared) reflectance in the falsecolor image greater than 0.29 on a 0-1 scale
4. Cloud fraction based on the IFT cloud mask.

The filter function accepts a list of tiles as input and also includes the option to reject tiles that have greater than a given
percentage of land pixels. It is recommended that the percentage be adjusted based on the tile size. Running the algorithm on 50 km
tiles appears to give reasonably good results but it is likely that the function parameters will be different for larger tiles.
"""

"""Compute the fraction of non-ocean pixels covered by cloud"""
function ocean_cloud_fraction(cloudmask, landmask)
    lm = vec(landmask)
    cf = vec(cloudmask)[.!lm]
    length(cf) > 0 ? (return round(mean(cf), digits=3)) : (return NaN)
end

"""Compute the contrast by differencing the 99th and 1st percentile"""
function robust_contrast(img_band, landmask; min_pct=1, max_pct=99)
    lm = vec(landmask)
    band_data = vec(img_band)[.!lm]
    length(band_data) > 0 ? (return round(percentile(band_data, max_pct) - percentile(band_data, min_pct), digits=3)) : (return 0)
end

"""Compute the fraction of non-cloudy ocean pixels with band 2 brightness above a given threshold"""
function possible_clear_sky_sea_ice(band_2, cloudmask, landmask; possible_ice_threshold=75/255)
    ocean_nocloud = vec(.! (landmask .| cloudmask))
    b2 = vec(band_2)[ocean_nocloud]
    length(b2) > 0 ? (return round(mean(b2 .> possible_ice_threshold), digits=3)) : (return 0)
end

function fitted_log_function(truecolor_band_1_contrast,
                             falsecolor_band_7_contrast,
                             cloud_fraction,
                             possible_sea_ice_fraction;
    parameters = [-1.25044852, 6.817628, -3.874728, -2.923836, 2.974168])
    X = [1, truecolor_band_1_contrast, falsecolor_band_7_contrast, cloud_fraction, possible_sea_ice_fraction]
    
    return 1 / (1 + exp(-sum(X .* parameters)))
end

function filter_function(true_color, false_color, cloud_mask, land_mask; land_thresh=0.9, prob_thresh=0.5)
    land_frac = mean(land_mask)
    land_frac > land_thresh && (return false)
    
    b1_contrast = robust_contrast(red.(true_color), land_mask)
    b7_contrast = robust_contrast(red.(false_color), land_mask)
    cloudfrac = ocean_cloud_fraction(cloud_mask, land_mask)
    poss_ice = possible_clear_sky_sea_ice(green.(false_color), cloud_mask, land_mask)

    p = fitted_log_function(b1_contrast, b7_contrast, cloudfrac, poss_ice)
    return p > prob_thresh
end