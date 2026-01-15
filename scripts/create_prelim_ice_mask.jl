using Pkg
Pkg.activate("cal-val")
using IceFloeTracker
using IceFloeTracker: AbstractCloudMaskAlgorithm, fill_holes
using Images
using Dates

save_loc = "../data/ift_prelim_ice_mask/"
dataset = filter(c -> c.visible_sea_ice == "yes", Watkins2026Dataset())
new_cmask = Watkins2025CloudMask();
b1_ice_min = 75/255

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
    edges, bincounts = build_histogram(banddata, 64; minval=0, maxval=1)

    ice_peak = IceFloeTracker.get_ice_peaks(edges, bincounts;
        possible_ice_threshold=b1_ice_min,
        minimum_prominence=0.01,
        window=3)

    thresh = 0.5 * (b1_ice_min + ice_peak)
    peaks_result = banddata .> thresh
    
    save(joinpath(save_loc, name*"-prelim_icemask.png"), Gray.(peaks_result))
end