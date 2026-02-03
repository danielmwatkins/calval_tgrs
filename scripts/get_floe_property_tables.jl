"""Access the matched pairs from the validation dataset and compute similarity metrics"""

using Pkg
Pkg.activate("../scripts/cal-val")

using IceFloeTracker
using DataFrames, Images, Dates, Statistics
import Images.KernelFactors: sobel

saveloc = "../data/floe_property_tables/"

min_area = 50
properties = ["label", "area", "bbox", "centroid", "convex_area", "major_axis_length", "minor_axis_length",
              "orientation", "perimeter"]

# Update with final column names
column_order = [:label, :row_centroid, :col_centroid,:min_row, :min_col, :max_row, :max_col,
                :area, :convex_area, :major_axis_length, :minor_axis_length, :orientation,
                :perimeter, :circularity, :solidity, :cloud_fraction_ift, :ice_fraction_ift,
                :tc_band_1_reflectance, :tc_band_4_reflectance, :tc_band_3_reflectance, 
                :fc_band_7_reflectance, :fc_band_2_reflectance, :fc_band_1_reflectance,
                :proc_grayscale_std, :tc_band_1_std, :boundary_gradient, :interior_gradient]
# tbd: multi band standard deviation of reflectance. currently just using the single band.

DWCM = IceFloeTracker.Watkins2025CloudMask()

### Replace when function added to IFT library
function Preprocess(tc_img, cloud_mask, land_mask)
    pmd_diff = PeronaMalikDiffusion(0.1, 0.1, 5, "exponential") 
    eq_clip = 0.01
    eq_nbins = 255
    rpix = 250 # arguments for kernel size
    cpix = 250 
    morph_strel = strel_diamond((5,5))
    min_area_init = 10

    # step 1: diffusion and sharpening
    begin
        proc_img = deepcopy(Gray.(tc_img)) # Cast truecolor to grayscale
        proc_img .= nonlinear_diffusion(proc_img, pmd_diff) |> unsharp_mask               
        apply_cloudmask!(proc_img, cloud_mask)
        apply_landmask!(proc_img, land_mask)
    end

    # step 2: adaptive histogram equalization
    begin
        equalized_gray = Gray.(IceFloeTracker.skimage.sk_exposure.equalize_adapthist(
            IceFloeTracker.ImageUtils.to_uint8(Float64.(proc_img) .* 255);
            kernel_size=(rpix, cpix), # Using default: image size divided by 8.
            clip_limit=eq_clip,  # Equivalent to MATLAB's 'ClipLimit'
            nbins=eq_nbins,      # Number of histogram bins. 255 is used to match the default in MATLAB script
        ))
        apply_cloudmask!(proc_img, cloud_mask)
        apply_landmask!(proc_img, land_mask)
    end
    return proc_img
end

### Remove when added to the IFT library
# Preliminary ice mask function using the red channel histogram to find mask threshold.
function prelim_icemask(tc_img_, cloud_mask, land_mask; b1_ice_min = 75/255)
    tc_img = RGB.(tc_img_)
    apply_cloudmask!(tc_img, cloud_mask)    
    apply_landmask!(tc_img, land_mask .> 0)
    banddata = red.(tc_img)
    edges, bincounts = build_histogram(banddata, 64; minval=0, maxval=1)
    ice_peak = IceFloeTracker.get_ice_peaks(edges, bincounts;
        possible_ice_threshold=b1_ice_min,
        minimum_prominence=0.01,
        window=3)
    thresh = 0.5 * (b1_ice_min + ice_peak)
    return banddata .> thresh
end

function imgradient_mag_sobel(img)
    Gy, Gx = imgradients(img, sobel, "replicate")
    return hypot.(Gx, Gy)
end


# component standard deviation
function component_standard_deviation(indexmap, img)
    c_idx = component_indices(indexmap)
    return Dict(l => std(vec(Float64.(img[c_idx[l]]))) for l in keys(c_idx))
end

# multi-band standard deviation?
# boundary statistics
function component_boundary_mean(indexmap, img, strel)
    labels = unique(indexmap)
    results = Dict()
    for l in labels
        idx = indexmap .== l
        bdry = dilate(idx, strel) .- idx
        results[l] = mean(vec(img[bdry .> 0]))
    end
    return results
end

dataset = Watkins2026Dataset()
cases = filter(f -> f.visible_floes == "yes", dataset)

for case in cases
    truecolor_image = modis_truecolor(case)
    falsecolor_image = modis_falsecolor(case)
    labeled_image = labels_map(validated_labeled_floes(case))
    landmask = modis_landmask(case) .> 0

    case_number = lpad(case.info.case_number, 3, "0")
    region = case.info.region
    satellite = case.info.satellite
    date = Dates.format(case.info.pass_time, DateFormat("YYYYmmdd"))
    suffix = "floe_properties.csv"

    if maximum(labeled_image) > 0
        df = IceFloeTracker.regionprops_table(labeled_image; properties=properties)
        df = filter(:area => a -> a .> min_area, df)
      
        if nrow(df) > 0
            df[!, :case_number] .= Int64(case.info.case_number)
            df[!, :circularity] .= 4 .* pi .* df[!, :area] ./ (df[!, :perimeter] .^ 2)
            df[!, :solidity] .= df[!, :area] ./ (df[!, :convex_area])
              
            # Get segment means
            cloudmask = create_cloudmask(falsecolor_image, DWCM)
            icemask =  prelim_icemask(truecolor_image, cloudmask, landmask)
            tc_image_data = SegmentedImage(truecolor_image, labeled_image)
            fc_image_data = SegmentedImage(falsecolor_image, labeled_image)
            cm_image_data = SegmentedImage(cloudmask, labeled_image)
            im_image_data = SegmentedImage(icemask, labeled_image)
    
            proc_img = float64.(Preprocess(truecolor_image, cloudmask, landmask))
            grad_mag = imgradient_mag_sobel(proc_img)
            gradient_data = SegmentedImage(grad_mag, labeled_image)
            
            # Add segment means to the dataframe
            # Q: .= or = when creating column?
            df[!,:cloud_fraction_ift] .= segment_mean.([cm_image_data], df[!, :label])
            df[!,:ice_fraction_ift] .= segment_mean.([im_image_data], df[!, :label])
    
            tc_means = segment_mean.([tc_image_data], df[!, :label])
            df[!,:tc_band_1_reflectance] .= red.(tc_means)
            df[!,:tc_band_4_reflectance] .= green.(tc_means)
            df[!,:tc_band_3_reflectance] .= blue.(tc_means)
    
            fc_means = segment_mean.([fc_image_data], df[!, :label])
            df[!,:fc_band_7_reflectance] .= red.(fc_means)
            df[!,:fc_band_2_reflectance] .= green.(fc_means)
            df[!,:fc_band_1_reflectance] .= blue.(fc_means)
    
            # Standard deviations
    
            cstdev = component_standard_deviation(labeled_image, proc_img)
            df[!,:proc_grayscale_std] .= [cstdev[l] for l in df[!, :label]]
    
            red_stdev = component_standard_deviation(labeled_image, red.(truecolor_image))
            df[!,:tc_band_1_std] .= [red_stdev[l] for l in df[!, :label]]
            
            # Boundary data
            bdry_grad = component_boundary_mean(labeled_image, grad_mag, strel_diamond((5,5)))
            df[!,:boundary_gradient] .= [bdry_grad[l] for l in df[!, :label]]
    
            # Interior gradient
            df[!,:interior_gradient] .= segment_mean.([gradient_data], df[!, :label])
            
            
            df = round.(df, digits=3)
            df[!,:area] = convert.(Int64,df[!,:area])
    
            # update column order
            select!(df, column_order)
            
            # update filename
            fpath = joinpath(saveloc, satellite, join([case_number, region, date, satellite, suffix], "-"))
            
            # save results
            df |> save(fpath)
        end
    end
end