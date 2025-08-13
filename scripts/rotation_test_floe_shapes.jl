"""Rotation experiment for calibration paper. All floes rotated by set amounts, then the differences are calculated. For the absdiffratio, note that to match the new paper, you need to multiply by 0.5"""

using Pkg;
Pkg.activate("cal-val")

using IceFloeTracker
using IceFloeTracker: load, regionprops_table, label_components, imshow, absdiffmeanratio, mismatch, addfloemasks!
using DataFrames, CSV, Interpolations, Images

test_images_loc = "/Users/dwatkin2/Documents/research/manuscripts/cal-val_ice_floe_tracker/ice_floe_validation_dataset/data/validation_dataset/binary_floes/"

# convenience functions
greaterthan0(x) = x .> 0 # convert labeled image to boolean
greaterthan05(x) = x .> 0.5 # used for the image resize step
imrotate_bin(x, r) = greaterthan05(collect(imrotate(x, deg2rad(r), method=BSpline(Constant()))))

# expose the non-normalized mismatch from the IFT
function mismatch_temp(fixed::AbstractArray, moving::AbstractArray, test_angles::AbstractArray)
    shape_differences = IceFloeTracker.shape_difference_rotation(
        fixed, moving, test_angles; imrotate_function=IceFloeTracker.imrotate_bin_clockwise_degrees
    )
    best_match = argmin((x) -> x.shape_difference, shape_differences)
    rotation_degrees = best_match.angle
    normalized_area = (sum(fixed) + sum(moving)) / 2
    normalized_mismatch = best_match.shape_difference / normalized_area
    return (mm=normalized_mismatch, rot=rotation_degrees, sd=best_match.shape_difference)
end

files = readdir(test_images_loc)
files = [f for f in files if occursin("aqua", f)]

# build psi fails for: 111, 130.
# files_subset = [f for f in files if parse(Int64, f[1:3]) != 130 && parse(Int64, f[1:3]) != 111]
for fname in files
    image = load(joinpath(test_images_loc, fname))
        
    # Add labels and get region properties
    labeled_image = label_components(image);
    # properties=["area", "convex_area", "perimeter", "major_axis_length", "minor_axis_length"]
    props = regionprops_table(labeled_image);
    
    addfloemasks!(props, greaterthan0.(labeled_image));
    
    df = DataFrame(
                   floe_id=Int16[],
                   rotation=Float64[],
                   area=Float64[],
                   convex_area=Float64[],
                   major_axis_length=Float64[],
                   minor_axis_length=Float64[],
                   perimeter=Float64[],
                   adr_area=Float64[],
                   adr_convex_area=Float64[],
                   adr_major_axis_length=Float64[],
                   adr_minor_axis_length=Float64[],
                   rotation_estimated=Float64[],
                   minimum_shape_difference=Float64[],
                   psi_s_correlation=Float64[],
                   )
    floe_id = 1
    for floe_data in eachrow(props)
        if floe_data["area"] >= 50
            im_init = copy(floe_data["mask"])
            
            # pad the floe to avoid changing floe area relative to image size
            n = Int64(round(maximum(size(im_init))))
            im_padded = collect(padarray(im_init, Fill(0, (n, n), (n, n))))
            
            # recalculate properties -- should be same as init except centroid/bbox
            init_props = regionprops_table(label_components(im_padded))
            for rotation in range(-45, 45, 31)
                im_rotated = imrotate_bin(im_padded, rotation)
                rotated_props = regionprops_table(label_components(im_rotated))
    
                normalized_mismatch, rotation_degrees, shape_difference = mismatch_temp(
                    im_init, im_rotated, -90:1:90)
                try
                    _psi = IceFloeTracker.buildψs.([im_init, im_rotated])
                    global r = round(IceFloeTracker.corr(_psi...), digits=3)
                
                catch e
                    @warn "Build Psi-S failed: $e"
                    global r = NaN
                end
                
                push!(df, (floe_id,
                           rotation,
                           rotated_props[1,:area],
                           rotated_props[1,:convex_area],
                           rotated_props[1,:major_axis_length],
                           rotated_props[1,:minor_axis_length],
                           rotated_props[1,:perimeter],
                           0.5*absdiffmeanratio(floe_data["area"], rotated_props[1,:area]),
                           0.5*absdiffmeanratio(floe_data["convex_area"], rotated_props[1,:convex_area]),
                           0.5*absdiffmeanratio(floe_data["major_axis_length"], rotated_props[1,:major_axis_length]),
                           0.5*absdiffmeanratio(floe_data["minor_axis_length"], rotated_props[1,:minor_axis_length]),
                           rotation_degrees,
                           shape_difference,
                           r
                           )) 
            end
        end
        floe_id += 1
    end
    
    
    print("Writing to file\n")
    CSV.write("../data/rotation_test/"*fname[1:3]*"-shape-rotation.csv", df);
end