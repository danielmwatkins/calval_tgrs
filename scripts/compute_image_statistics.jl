using Pkg
Pkg.activate("../notebooks/calval")
using IceFloeTracker
using Images
using CSV
using DataFrames
using StatsBase

include("dev/filter_tiles.jl")

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


n = 98
dataloc = "../data/validation_dataset/modis_500km/"
saveloc = "25km_data_tables"

tc_filenames = filter(x -> !occursin(".DS", x), readdir(joinpath(dataloc, "truecolor")))

function get_image_stats(tc_image, fc_image, cloud_mask, land_mask, tile_markers, tiles)
    df = DataFrame(
                   tile_index=Int16[],
                   ground_truth=[],
                   cloud_fraction=Float64[],                   
                   land_fraction=Float64[],
                   truecolor_band_1_contrast=Float64[],
                   falsecolor_band_2_contrast=Float64[],
                   falsecolor_band_7_contrast=Float64[],
                   truecolor_band_1_entropy=Float64[],
                   falsecolor_band_2_entropy=Float64[],
                   falsecolor_band_7_entropy=Float64[],
                   possible_sea_ice_fraction=Float64[]
                   )
    
    for (index, tile) in enumerate(tiles)
        push!(df, (
                   tile_index = index,
                   ground_truth = maximum(tile_markers[tile...]) > 0,
                   cloud_fraction = ocean_cloud_fraction(cloud_mask[tile...], land_mask[tile...]),
                   land_fraction = mean(land_mask[tile...]),
                   truecolor_band_1_contrast = robust_contrast(red.(tc_image[tile...]), land_mask[tile...]),
                   falsecolor_band_2_contrast = robust_contrast(green.(fc_image[tile...]), land_mask[tile...]),
                   falsecolor_band_7_contrast = robust_contrast(red.(fc_image[tile...]), land_mask[tile...]),
                   truecolor_band_1_entropy = Images.entropy(red.(tc_image[tile...]) .* .! land_mask[tile...]), 
                   falsecolor_band_2_entropy = Images.entropy(green.(fc_image[tile...]) .* .! land_mask[tile...]),
                   falsecolor_band_7_entropy = Images.entropy(red.(fc_image[tile...]) .* .! land_mask[tile...]),
                   possible_sea_ice_fraction = possible_clear_sky_sea_ice(
                                                    green.(fc_image[tile...]),
                                                    cloud_mask[tile...],
                                                    land_mask[tile...]))
                )
    end
    return df
end



for tc_filename in tc_filenames

    fc_filename = replace(tc_filename, "truecolor" => "falsecolor")
    markers_filename = replace(replace(tc_filename, "truecolor" => "markers"), "tiff" => "png")
    landmask_filename = occursin("aqua", tc_filename) ? replace(tc_filename, "aqua-" => "") : replace(tc_filename, "terra-" => "")
    landmask_filename = replace(landmask_filename, "truecolor" => "landmask")



    landmask_filename = occursin("aqua", tc_filename) ? replace(tc_filename, "aqua-" => "") : replace(tc_filename, "terra-" => "")
    landmask_filename = replace(landmask_filename, "truecolor" => "landmask")
    
    tc_image = RGB.(load(joinpath(dataloc, "truecolor", tc_filename)))
    fc_image = RGB.(load(joinpath(dataloc, "falsecolor", replace(tc_filename, "truecolor" => "falsecolor"))))
    tile_markers = Gray.(load(joinpath(dataloc, "tile_markers", markers_filename))) .> 0
    land_image = Gray.(load(joinpath(dataloc, "landmask", landmask_filename)))
    
    cloud_mask = IceFloeTracker.create_cloudmask(fc_image, cmask)
    land_mask = land_image .> 0.1
    
    # with 256 m pixels instead of our usual 250 m tiles, we have approx 98 pixels per 25 km.
    tiles = IceFloeTracker.get_tiles(tc_image, n)
    
    # use the markers to make a checkerboard
    for tile in filter(t -> maximum(tile_markers[t...]) .> 0, tiles)
        tile_markers[tile...] .= 1
    end

    df = get_image_stats(tc_image, fc_image, cloud_mask, land_mask, tile_markers, tiles)
    
    CSV.write(joinpath(dataloc, saveloc, replace(tc_filename, "truecolor.tiff" => "statistics.csv")), df)
end