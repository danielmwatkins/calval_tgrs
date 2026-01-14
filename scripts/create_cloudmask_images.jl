"""
Loop through the validation dataset, creating a cloud mask for each case. Generates the original LA2019 mask, an LA2019 mask with updated parameters, and the new W25 mask with the morphological cleanup to remove speckle.
"""

using Pkg
Pkg.activate("cal-val")
using IceFloeTracker
using IceFloeTracker: AbstractCloudMaskAlgorithm, fill_holes
using Images
using Dates

save_loc = "../data/ift_cloud_mask/"
dataset = Watkins2026Dataset()

cloud_mask_settings = (
    prelim_threshold=53.0/255.,
    band_7_threshold=130.0/255.,
    band_2_threshold=170.0/255.,
    ratio_lower=0.0,
    ratio_offset=0.0,
    ratio_upper=0.52
)

old_cmask = LopezAcostaCloudMask();
new_init = LopezAcostaCloudMask(cloud_mask_settings...);
new_cmask = Watkins2025CloudMask();

for case in dataset
    cn = lpad(case.info[:case_number], 3, "0")
    region = case.info[:region]
    date = Dates.format(case.info[:pass_time], "YYYYmmdd")
    sat = case.info[:satellite]
    name = join([cn, region, "100km", date, sat, "250m"], "-")
    
    fc = RGB.(modis_falsecolor(case))
    lm = Gray.(modis_landmask(case)) .> 0
    old = old_cmask(fc)
    new_raw = new_init(fc)
    new_clean = new_cmask(fc)
    save(joinpath(save_loc, "lopez_acosta", name*"-cloudmask.png"), old)
    save(joinpath(save_loc, "initial", name*"-cloudmask.png"), new_raw)
    save(joinpath(save_loc, "cleaned", name*"-cloudmask.png"), new_clean)
end