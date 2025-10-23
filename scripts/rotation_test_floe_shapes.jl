"""Rotation experiment for calibration paper. All floes rotated by set amounts, then the differences are calculated. For the absdiffratio, note that to match the new paper, you need to multiply by 0.5 to convert from |x - y|/|mean(x, y)| into |x - y| / |x + y|."""

using Pkg
Pkg.activate("cal-val")
using IceFloeTracker
using IceFloeTracker: Watkins2025GitHub
using IceFloeTracker.Tracking: absdiffmeanratio, buildψs, corr, mismatch
using DataFrames
using Interpolations
using CSVFiles

# The "ref" parameter points to the newest version of the dataset.
# If any scripts have been run with this already, it will be faster due to the data already being downloaded.
data_loader = Watkins2025GitHub(; ref="1d36f82a5cb337839ab039bffca8961a97241c22")
data_set = data_loader(c -> c.visible_floes == "yes")

# simple utility function to turn array to bitmap
greaterthan05(x) = x .> 0.5

# and to rotate an image while preventing cropping
imrotate_bin_nocrop(x, r) = greaterthan05(collect(imrotate(x, r; method=BSpline(Constant()))))

for (case, metadata) in zip(data_set.data, eachrow(data_set.metadata))
    print("Running case ", metadata.case_number, " ", metadata.satellite, "\n")

    labeled_image = case.validated_labeled_floes

    props = regionprops_table(labeled_image.image_indexmap)
    
    # bug in addfloemasks requires that the dataframe not have extra rows
    addfloemasks!(props, greaterthan05.(labeled_image.image_indexmap))
    
    # hence we have to add them in after
    props_labeled = regionprops_table(labeled_image.image_indexmap; properties=["label"])
    props[:, :label] = props_labeled[:, :label]

    # initialize the dataframe
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
                   normalized_shape_difference=Float64[],
                   psi_s_correlation=Float64[],
                   )
    
    for floe_data in eachrow(props)
        if floe_data["area"] >= 50 && floe_data["area"] <= 350^2 # had some errors where the background was treated as a floe
            
            for rotation in range(-45, 45, 31)
                im_rotated = imrotate_bin_nocrop(floe_data[:mask], deg2rad(rotation))
                rotated_props = regionprops_table(label_components(im_rotated))
                # Search the 90-degree window surrounding the true rotation. 
                test_angles = range(; start=rotation - 45, stop=rotation + 45, step=1)
                normalized_shape_difference, rotation_estimated = mismatch(floe_data[:mask], im_rotated, test_angles)
    
                try
                    _psi = buildψs.([floe_data[:mask], im_rotated])
                    global r = round(corr(_psi...), digits=3)
                
                catch e
                    @warn "Build Psi-S failed: $e"
                    global r = NaN
                end
                
                push!(df, (floe_data[:label],
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
                           rotation_estimated,
                           normalized_shape_difference,
                           r
                           )) 
            end
        end
    end    
    
    print("Writing to file\n")
    
    df |> save("../data/rotation_test/"*case.name*"-shape-rotation.csv")
end