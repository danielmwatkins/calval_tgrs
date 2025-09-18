using Pkg
Pkg.activate("../notebooks/calval")
using IceFloeTracker
using Images
using CSV
using DataFrames
using StatsBase

include("dev/validation_data.jl")

data_loader = Watkins2025GitHub(; ref="a451cd5e62a10309a9640fbbe6b32a236fcebc70")
dataset = data_loader(c -> c.case_number <= 188)

"""Compute the fraction of non-ocean pixels covered by cloud"""
function ocean_cloud_fraction(cloudfrac, landmask)
    lm = vec(landmask)
    cf = vec(cloudfrac)[.!lm]
    length(cf) > 0 ? (return round(mean(cf), digits=3)) : (return missing)
end

"""Compute the contrast by differencing the 99th and 1st percentile"""
function robust_contrast(img_band, landmask; min_pct=1, max_pct=99)
    lm = vec(landmask)
    band_data = vec(img_band)[.!lm]
    length(band_data) > 0 ? (return round(percentile(band_data, max_pct) - percentile(band_data, min_pct), digits=3)) : (return NaN)
end

function possible_sea_ice(band_2, landmask, cloudmask; possible_ice_threshold=75/255)
    ocean_nocloud = vec(.! (landmask .| cloudmask))
    b2 = vec(band_2)[ocean_nocloud]
    length(b2) > 0 ? (return round(mean(b2 .> possible_ice_threshold), digits=3)) : (return NaN)
end


# Set up cloud mask
cloud_mask_settings = (
    prelim_threshold=53.0/255.,
    band_7_threshold=130.0/255.,
    band_2_threshold=169.0/255.,
    ratio_lower=0.0,
    ratio_offset=0.0,
    ratio_upper=0.53
)
cmask = LopezAcostaCloudMask(cloud_mask_settings...)

entropy_data = []
contrast_data = []
cf_data = []
poss_ice_data = []
case_names = []
satellites = []
numbers = []
landfraction = []
tile_num = []
for case in dataset
    if occursin("aqua", case.name)
        print(case.name, "\n")
    end
   
    tc_img = RGB.(case.modis_truecolor)
    fc_img =  RGB.(case.modis_falsecolor)
    cloudmask = Gray.(IceFloeTracker.create_cloudmask(fc_img, cmask)) .> 0
    landmask = Gray.(case.modis_landmask) .> 0

    tiles = IceFloeTracker.get_tiles(tc_img, 200)

    append!(tile_num, 0)
    append!(entropy_data, Images.entropy(tc_img .* (.!landmask)))
    append!(contrast_data, robust_contrast(red.(tc_img), landmask))
    append!(cf_data, ocean_cloud_fraction(cloudmask, landmask))
    append!(poss_ice_data, possible_sea_ice(green.(fc_img), landmask, cloudmask))
    append!(numbers, [case.metadata[:case_number]])
    append!(case_names, [case.name])
    append!(satellites, [case.metadata[:satellite]])

    append!(landfraction, round(mean(landmask), digits=2))

    for (idx, tile) in enumerate(tiles)
        append!(tile_num, idx)
        append!(entropy_data, Images.entropy(tc_img[tile...] .* (.!landmask[tile...])))
        append!(contrast_data, robust_contrast(red.(tc_img[tile...]), landmask[tile...]))
        append!(cf_data, ocean_cloud_fraction(cloudmask[tile...], landmask[tile...]))
        append!(poss_ice_data, possible_sea_ice(green.(fc_img[tile...]), landmask[tile...], cloudmask[tile...]))
        append!(numbers, [case.metadata[:case_number]])
        append!(case_names, [case.name])
        append!(satellites, [case.metadata[:satellite]])
        append!(landfraction, round(mean(landmask[tile...]), digits=2)) 
    end
end

df = DataFrame(case_number = numbers,
           case_name = case_names,
           satellite = satellites,
           tile = tile_num,
           entropy = entropy_data,
           contrast_red = contrast_data,
           ocean_cloud_fraction = cf_data,
           possible_clear_sky_ice = poss_ice_data,
           land_fraction = landfraction)
CSV.write("../data/validation_dataset_image_statistics.csv", df)

