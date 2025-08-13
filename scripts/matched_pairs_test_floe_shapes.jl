"""Access the matched pairs from the validation dataset and compute similarity metrics"""

using Pkg;
Pkg.activate("cal-val")

using IceFloeTracker
using IceFloeTracker: load, regionprops_table, label_components, imshow, absdiffmeanratio, mismatch, addfloemasks!
using DataFrames, CSV, Interpolations, Images

ice_floe_database_loc =  "/Users/dwatkin2/Documents/research/manuscripts/cal-val_ice_floe_tracker/ice_floe_validation_dataset/"

matched_pairs_tables_loc = joinpath(ice_floe_database_loc, "data/validation_dataset/property_tables/matched")
test_images_loc = joinpath(ice_floe_database_loc, "data/validation_dataset/labeled_floes/")


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
    
# load information on pairs
# each of these is a table with paired labels for images
files = [f for f in readdir(matched_pairs_tables_loc) if occursin(".csv", f)]; 

for file in files
    case = split(file, "-")[1]
    df_pairs = DataFrame(CSV.File(joinpath(matched_pairs_tables_loc, file)))
    print(file, "\n")
    # First check for whether there is a match at all
    if size(df_pairs)[1] > 0 && !ismissing(df_pairs[1, :aqua_label])
        
        # Load the labeled image and convert to an integer-valued Matrix
        lb_aqua = channelview(Int64.(
                            load(
                                joinpath(test_images_loc, replace(file, "matched-floe_properties.csv" => "aqua-labeled_floes.tiff")))
                                )
                            )
        lb_terra = channelview(Int64.(
                            load(
                                joinpath(test_images_loc, replace(file, "matched-floe_properties.csv" => "terra-labeled_floes.tiff")))
                                )
                            )
    
        # Retrieve region props and add floe masks
        proplist = ["bbox", "centroid", "label", "area", "convex_area",
            "perimeter", "major_axis_length", "minor_axis_length"]
        
        props_aqua = regionprops_table(lb_aqua);
        props_terra = regionprops_table(lb_terra);
        addfloemasks!(props_aqua, greaterthan0.(lb_aqua));
        addfloemasks!(props_terra, greaterthan0.(lb_terra));
    
        # Bug in addfloemasks means we have to merge label back in
        props_labels_aqua = regionprops_table(lb_aqua, properties=["area", "label", "perimeter_crofton", "bbox"])
        props_labels_terra = regionprops_table(lb_terra, properties=["area", "label", "perimeter_crofton", "bbox"])
        props_aqua = innerjoin(props_aqua, props_labels_aqua, on=[:area, :min_row, :max_row, :min_col, :max_col])
        props_terra = innerjoin(props_terra, props_labels_terra, on=[:area, :min_row, :max_row, :min_col, :max_col])
        
        # Initialize dataframe for the shape comparison
        df = DataFrame(
                   aqua_label=Int64[],
                   terra_label=Int64[],               
                   aqua_area=Float64[],
                   aqua_convex_area=Float64[],
                   aqua_major_axis_length=Float64[],
                   aqua_minor_axis_length=Float64[],
                   aqua_perimeter=Float64[],
                   aqua_perimeter_crofton=Float64[],
                   terra_area=Float64[],
                   terra_convex_area=Float64[],
                   terra_major_axis_length=Float64[],
                   terra_minor_axis_length=Float64[],
                   terra_perimeter=Float64[],
                   terra_perimeter_crofton=Float64[],
                   adr_area=Float64[],
                   adr_convex_area=Float64[],
                   adr_major_axis_length=Float64[],
                   adr_minor_axis_length=Float64[],
                   rotation_estimated=Float64[],
                   minimum_shape_difference=Float64[],
                   psi_s_correlation=Float64[],
                   )
    
        for matches in eachrow(df_pairs)
            global row_aqua = props_aqua[props_aqua.label .== matches.aqua_label, :]
            row_terra = props_terra[props_terra.label .== matches.terra_label, :]
    
            normalized_mismatch, rotation_degrees, shape_difference = mismatch_temp(row_aqua[1, :mask], row_terra[1, :mask], -45:1:45)
            # TBD: Also add dataframe to store the rotation vs SD vectors 
            
            try
                _psi = IceFloeTracker.buildψs.([row_aqua[1, :mask], row_terra[1, :mask]])
                global psi_s_correlation = round(IceFloeTracker.corr(_psi...), digits=3)
            
            catch e
                @warn "Build Psi-S failed: $e"
                global psi_s_correlation = NaN
            end
            
            push!(df, (matches.aqua_label,
                       matches.terra_label,
                       row_aqua[1, :area],
                       row_aqua[1, :convex_area],
                       row_aqua[1, :major_axis_length],
                       row_aqua[1, :minor_axis_length],
                       row_aqua[1, :perimeter],
                       row_aqua[1, :perimeter_crofton],
                       row_terra[1, :area],
                       row_terra[1, :convex_area],
                       row_terra[1, :major_axis_length],
                       row_terra[1, :minor_axis_length],
                       row_terra[1, :perimeter],     
                       row_terra[1, :perimeter_crofton],     
                       0.5*absdiffmeanratio(row_aqua[1, :area], row_terra[1, :area]),
                       0.5*absdiffmeanratio(row_aqua[1, :convex_area], row_terra[1, :convex_area]),
                       0.5*absdiffmeanratio(row_aqua[1, :major_axis_length], row_terra[1, :major_axis_length]),
                       0.5*absdiffmeanratio(row_aqua[1, :minor_axis_length], row_terra[1, :minor_axis_length]),
                       rotation_degrees,
                       shape_difference,
                       psi_s_correlation
                       )) 
        end
    
    print("Writing to file\n")
    CSV.write("../data/matched_pairs_test/"*case*"-matched_pairs.csv", df);
    end
end
