using Pkg
Pkg.activate("calval")
using IceFloeTracker
using IceFloeTracker: AbstractCloudMaskAlgorithm, fill_holes
using Images
using Dates

save_loc = "../data/ift_prelim_ice_mask/"
dataset = filter(c -> c.visible_sea_ice == "yes", Watkins2026Dataset())
new_cmask = Watkins2026CloudMask(band_7_threshold=0.16, band_2_threshold=0.34, opening_strel=strel_disk(3), dilation_strel=strel_disk(2));

b1_ice_min = 0.45

for case in dataset
    cn = lpad(case.info[:case_number], 3, "0")
    region = case.info[:region]
    date = Dates.format(case.info[:pass_time], "YYYYmmdd")
    sat = case.info[:satellite]
    name = join([cn, region, "100km", date, sat, "250m"], "-")
    
    tc_img = RGB.(modis_truecolor(case))
    fc_img = RGB.(modis_falsecolor(case))
    land_mask = Gray.(modis_landmask(case)) .> 0
    cloud_mask = new_cmask(fc_img)

    banddata = red.(apply_landmask(tc_img, land_mask)) .* .! cloud_mask
    edges, bincounts = build_histogram(banddata, 128; minval=0, maxval=1)

    ice_peak = IceFloeTracker.get_ice_peaks(edges, bincounts;
        possible_ice_threshold=b1_ice_min,
        minimum_prominence=0.01,
        window_size=3)

    thresh = 0.5 * (b1_ice_min + ice_peak)
    peaks_result = banddata .> thresh
    
    save(joinpath(save_loc, name*"-prelim_icemask.png"), Gray.(peaks_result))
end