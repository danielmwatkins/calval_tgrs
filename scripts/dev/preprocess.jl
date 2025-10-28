# New preprocessing routine. 
# Filters tiles
using IceFloeTracker: unsharp_mask, PeronaMalikDiffusion
using Images: adjust_histogram, AdaptiveEqualization, ContrastStretching, mreconstruct, strel_diamond

function preprocess_image(truecolor_image, cloud_mask, land_mask, tiles;
    maximum_land_fraction = 0.75,
    maximum_cloud_fraction = 0.9,
    adapt_histeq_clip = 0.99,
    cloudy_adapt_histeq_clip = 0.75,
    adapt_histeq_block_size = 200,
    contrast_stretching_slope = 3.5,
    reconstruct_se = strel_diamond((5, 5))
    )
    # Select a subset of the tiles to preprocess
    filtered_tiles = filter(
        t -> mean(land_mask.dilated[t...]) < maximum_land_fraction &&
                 mean(cloud_mask[t...]) < maximum_cloud_fraction, tiles)

    length(filtered_tiles) == 0 && return missing # what's the right type to return here?

    # Set up blocks for adaptive histogram equalization
    rblocks, cblocks = round.(Int64, size(tc_img) ./ adapt_histeq_block_size)
    rblocks = maximum((1, rblocks))
    cblocks = maximum((1, cblocks))

    enhanced_gray_init = Gray.(truecolor_image)
    enhanced_gray = deepcopy(enhanced_gray_init) .* 0
    
    cloudy_tiles_mask = zeros(size(enhanced_gray))

    # Can include this in the function settings
    pmd_diff = PeronaMalikDiffusion(0.1, 0.1, 5, "exponential")

    for tile in filtered_tiles
        enhanced_gray[tile...] .= nonlinear_diffusion(enhanced_gray_init[tile...], pmd_diff) |> unsharp_mask
        if mean(cloud_mask[tile...]) > 0.5
            cloudy_tiles_mask[tile...] .= mean(cloud_mask[tile...])
        end        
    end

    weights = Float64.(mapwindow(mean, Gray.(cloudy_tiles_mask), (51,51)))
    aggressive_equalization = adjust_histogram(enhanced_gray,
                AdaptiveEqualization(clip=0.75, rblocks=rblocks, cblocks=rblocks, nbins=256))
    enhanced_gray .= (1 .- weights) .* enhanced_gray .+ weights .* aggressive_equalization

    adjust_histogram!(enhanced_gray, ContrastStretching(slope=3.5));

    dilated_img = dilate(enhanced_gray, reconstruct_se)
    reconstructed_complement = mreconstruct(
        dilate, complement.(dilated_img), complement.(enhanced_gray), strel_diamond((3, 3)));
    
    return complement.(reconstructed_complement)
end