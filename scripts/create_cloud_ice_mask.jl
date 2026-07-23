using Pkg
Pkg.activate("calval")
using IceFloeTracker
using Images
using Dates

ice_save_loc = "../data/ift_ice_mask/"
cloud_save_loc = "../data/ift_cloud_mask/"
dataset = filter(c -> c.visible_sea_ice == "yes", Watkins2026Dataset())

get_cloud_mask = Watkins2026CloudMask(
    band_7_threshold=0.16,
    band_2_threshold=0.34,
    opening_strel=strel_disk(3),
    dilation_strel=strel_disk(2)
)
get_ice_mask = IceDetectionBrightnessMidpoint(; minimum_reflectance=0.45)

for case in dataset
    cn = lpad(case.info[:case_number], 3, "0")
    region = case.info[:region]
    date = Dates.format(case.info[:pass_time], "YYYYmmdd")
    sat = case.info[:satellite]
    name = join([cn, region, "100km", date, sat, "250m"], "-")
    
    fc_img = RGB.(modis_falsecolor(case))
    land_mask = Gray.(modis_landmask(case)) .> 0
    apply_landmask!(fc_img, land_mask)
    
    cloud_mask = get_cloud_mask(fc_img)
    apply_cloudmask!(fc_img, cloud_mask)

    ice_mask = get_ice_mask(Gray.(blue.(fc_img)))
    
    save(joinpath(ice_save_loc, name*"-ice_mask.png"), Gray.(ice_mask))
    save(joinpath(cloud_save_loc, name*"-cloud_mask.png"), Gray.(cloud_mask))
end
