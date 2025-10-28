# Tests the preprocessing results by calculating the variance and mean of the ice floes and the ice floe boundaries using the validated floe boundaries.
# Generates test images with enhanced_grayscale, morphological_grayscale, and 3-layer binarization results (overlaid Otsu and AdaptiveThreshold).

using Pkg
Pkg.activate("cal-val")
using IceFloeTracker

using IceFloeTracker: nonlinear_diffusion, Watkins2025CloudMask, fill_holes, Watkins2025GitHub
using IceFloeTracker.Filtering: PeronaMalikDiffusion
using StatsBase
using DataFrames
using CSVFiles
using Images

new_cmask = Watkins2025CloudMask();
pmd_diff = PeronaMalikDiffusion(0.1, 0.1, 5, "exponential")

df = DataFrame(
               case_number=Int16[],
               satellite=String[],
               init_ice_mean=Float64[],
               init_ice_stdv=Float64[],
               init_bdry_mean=Float64[],
               init_bdry_stdv=Float64[],
               proc_ice_mean=Float64[],                   
               proc_ice_stdv=Float64[],
               proc_bdry_mean=Float64[],
               proc_bdry_stdv=Float64[],
               morph_ice_mean=Float64[],
               morph_ice_stdv=Float64[],
               morph_bdry_mean=Float64[],
               morph_bdry_stdv=Float64[]
               )  


data_loader = Watkins2025GitHub(; ref="1d36f82a5cb337839ab039bffca8961a97241c22")
dataset = data_loader(c -> c.case_number <= 189 && c.visible_floes == "yes");
for (case, metadata) in zip(dataset.data, eachrow(dataset.metadata))
    
    tc_img = RGB.(case.modis_truecolor)
    fc_img = RGB.(case.modis_falsecolor)
    
    land_mask = IceFloeTracker.create_landmask(case.modis_landmask, strel_diamond((3,3)))
    cloud_mask = IceFloeTracker.create_cloudmask(fc_img, new_cmask)
    tc_img[land_mask.non_dilated] .= 0
    
    tile_size_pixels = 200 # here it's the *length* in pixels, not the 
    minimum_ocean_pixels = 100^2 # setting for the simple tile filter
    
    tiles = IceFloeTracker.get_tiles(tc_img, tile_size_pixels)
    
    # Get list of tiles to process:
    # just use a minimum value for land for now
    filtered_tiles = filter(
        t -> sum(.!land_mask.dilated[t...]) > minimum_ocean_pixels, tiles)
    
    # @info "Diffuse and sharpen the full image"
    tc_diffused_sharpened = deepcopy(tc_img)
    cloudy_tiles_mask = zeros(size(tc_diffused_sharpened))
    img = deepcopy(tc_img)
    for tile in filtered_tiles
            tc_diffused_sharpened[tile...] .= nonlinear_diffusion(img[tile...], pmd_diff) |> IceFloeTracker.unsharp_mask
        
            mean(cloud_mask[tile...]) > 0.75 && 
                begin
                    cloudy_tiles_mask[tile...] .= mean(cloud_mask[tile...])
                end
    end
    
    # @info "Histogram adjustment"
    rblocks, cblocks = round.(Int64, size(tc_img) ./ tile_size_pixels)
    rblocks = maximum((1, rblocks))
    cblocks = maximum((1, cblocks))
    # Q1: Do we have better results with or without AdaptiveEqualization?
    adjust_histogram!(tc_diffused_sharpened, AdaptiveEqualization(clip=0.99, rblocks=rblocks, cblocks=cblocks, nbins=256))
    
    aggressive_equalization = adjust_histogram(tc_diffused_sharpened, AdaptiveEqualization(clip=0.75, rblocks=rblocks, cblocks=rblocks, nbins=256))

    # Weighted average. TBD: Use a rolling window instead
    tc_diffused_sharpened = (ones(size(tc_diffused_sharpened)) .- cloudy_tiles_mask) .* tc_diffused_sharpened .+ cloudy_tiles_mask .* aggressive_equalization

    # Contrast stretching
    enhanced_gray = Gray.(tc_diffused_sharpened) # .* .! cloud_mask
    # Increase the range from dark to bright
    adjust_histogram!(enhanced_gray, ContrastStretching(slope=3.5));
    
    #  Grayscale morphological operations
    dilated_img = dilate(enhanced_gray, strel_diamond((5, 5)))
    reconstructed_complement = mreconstruct(
        dilate, complement.(dilated_img), complement.(enhanced_gray), strel_diamond((3, 3)));

    # Temp. visualization with binarization. Note that it has errors where the original is all blank, 
    # so in the final version we would need to 
    three_layer_from_gray = 0.5 .* binarize(complement.(enhanced_gray), Otsu()) .+
                            0.5 .* binarize(complement.(enhanced_gray), AdaptiveThreshold(;window_size=tile_size_pixels, percentage=0))
    three_layer_from_reconstruct = 0.5 .* binarize(reconstructed_complement, Otsu()) .+
                                    0.5 .* binarize(reconstructed_complement, AdaptiveThreshold(;window_size=tile_size_pixels, percentage=0))
    save("../figures/preprocess_testing/"*case.name*".png",  mosaicview(enhanced_gray, reconstructed_complement, three_layer_from_gray, three_layer_from_reconstruct, nrow=1))
    
    
    ice_index = vec(case.validated_binary_floes .> 0)
    ice_mean = mean(Float64.(vec(Gray.(case.modis_truecolor)))[ice_index])
    ice_stdv = std(Float64.(vec(Gray.(case.modis_truecolor)))[ice_index])
    
    bdry = dilate(case.validated_binary_floes .> 0, strel_diamond((5,5))) .- erode(case.validated_binary_floes .> 0, strel_diamond((5,5)))
    bdry_index = vec(bdry .> 0)
    bdry_mean = mean(Float64.(vec(Gray.(case.modis_truecolor)))[bdry_index])
    bdry_stdv = std(Float64.(vec(Gray.(case.modis_truecolor)))[bdry_index])
    
    enhanced_ice_mean = mean(Float64.(vec(enhanced_gray)[ice_index]))
    enhanced_ice_stdv = std(Float64.(vec(enhanced_gray)[ice_index]))
    enhanced_bdry_mean = mean(Float64.(vec(enhanced_gray)[bdry_index]))
    enhanced_bdry_stdv = std(Float64.(vec(enhanced_gray)[bdry_index]))
    morphed = complement.(reconstructed_complement)
    morph_ice_mean = mean(Float64.(vec(morphed)[ice_index]))
    morph_ice_stdv = std(Float64.(vec(morphed)[ice_index]))
    morph_bdry_mean = mean(Float64.(vec(morphed)[bdry_index]))
    morph_bdry_stdv = std(Float64.(vec(morphed)[bdry_index]))
    
    push!(df, [metadata[:case_number], metadata[:satellite],
            ice_mean, ice_stdv, bdry_mean, bdry_stdv,
            enhanced_ice_mean, enhanced_ice_stdv, enhanced_bdry_mean, enhanced_bdry_stdv,
            morph_ice_mean, morph_ice_stdv, morph_bdry_mean, morph_bdry_stdv])
end

df |> save("../data/preprocess-results.csv")