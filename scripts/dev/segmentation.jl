#### Functions in the FSPipeline, placed here for early access ####

"""
    extended_regionprops_table()

Calls @ref[`regionprops_table`] with the provided `properties` list. Then, adds information on
floe-average overlap with the provided `masks` (expects Dict with mask name => binary mask), 
band-average reflectance from the falsecolor image, and band 1 boundary contrast. Finally, uses
a provided probability function to add a `probability` column indicating the likelihood the object
is an ice floe.

"""
function extended_regionprops_table(
    img_indexmap,
    falsecolor_image,
    masks;
    boundary_radius=15,
    properties = [
        :label, :area, :perimeter, :bbox,
        :centroid, :convex_area, :major_axis_length,
        :minor_axis_length, :orientation,
        :circularity, :solidity],
    probability_function=LogisticRegressionFilter,
    convex_area_algorithm=PolygonConvexArea(),
)
    img_indexmap = copy(img_indexmap)
    indices = component_indices(img_indexmap)

    props_df = regionprops_table(img_indexmap;
        properties=properties,
        convex_area_algorithm=convex_area_algorithm,
    )
    # Return empty dataframe if no floes in image
    nrow(props_df) == 0 && return props_df

    transform!(props_df, :area => ByRow(x -> x^0.5) => :length_scale)
    
    # Don't allow circularity or solidity greater than 1
    transform!(props_df, :solidity => ByRow(x -> minimum([x, 1])) => :solidity)
    transform!(props_df, :circularity => ByRow(x -> minimum([x, 1])) => :circularity)
    
    # Get the average area coverage for each of the masks
    mask_mean(r, mask) = mean(mask[indices[r]])
    for k in keys(masks)
        props_df[:, Symbol(k, "_fraction")] =  mask_mean.(props_df[:, :label], [masks[k]])
    end
    
    # Get the mean reflectance and mean boundary reflectance, and expand the results into named color channels
    add_mean_reflectance!(props_df, falsecolor_image, indices)
    add_mean_boundary_reflectance!(props_df, falsecolor_image, img_indexmap; radius=boundary_radius)
    
    # TODO: generalize with a map from channel number to channel name
    # Could make this a for loop with transform!()
    props_df[:, :b7_mean_reflectance] = red.(props_df.mean_reflectance)
    props_df[:, :b2_mean_reflectance] = green.(props_df.mean_reflectance)
    props_df[:, :b1_mean_reflectance] = blue.(props_df.mean_reflectance)

    props_df[:, :b7_mean_boundary_reflectance] = red.(props_df.mean_boundary_reflectance)
    props_df[:, :b2_mean_boundary_reflectance] = green.(props_df.mean_boundary_reflectance)
    props_df[:, :b1_mean_boundary_reflectance] = blue.(props_df.mean_boundary_reflectance)

    props_df[:, :b7_mean_boundary_contrast] = props_df[:, :b7_mean_reflectance] .- props_df[:, :b7_mean_boundary_reflectance]
    props_df[:, :b2_mean_boundary_contrast] = props_df[:, :b2_mean_reflectance] .- props_df[:, :b2_mean_boundary_reflectance]
    props_df[:, :b1_mean_boundary_contrast] = props_df[:, :b1_mean_reflectance] .- props_df[:, :b1_mean_boundary_reflectance]
    
    # TODO: generalize to include inplace option
    props_df[:, :probability] .= probability_function(props_df)

    # Drop the RGB columns in the returned dataframe
    return props_df[:, Not(:mean_reflectance, :mean_boundary_reflectance)]
end

"""LogisticRegressionFilter(df;
    coefs = Dict(
        "intercept"           => -97.1879,
        "length_scale"        => 0.1267,
        "solidity"            => 91.164,
        "b1_mean_reflectance" => 7.354,
        "b7_mean_reflectance" => -1.517,
        "b1_mean_boundary_contrast" => 2.239,
        )
    )
    LogisticRegressionFilter!(df; coefs)

Apply the logistic regression function with the provided set of coefficients. The in-place version
adds a column "probability" to the dataframe, while the non-in-place version returns a vector
with probabilities.

"""
function LogisticRegressionFilter(df;
    coefs = Dict(
        "intercept"                 => -97.1879,
        "length_scale"              => 0.1267,
        "solidity"                  => 91.164,
        "b1_mean_reflectance"       => 7.354,
        "b7_mean_reflectance"       => -1.517,
        "b1_mean_boundary_contrast" => 2.239,
        )
    )
    colnames = [x for x in keys(coefs)]
    b = [x for x in values(coefs)]
    df[:, :intercept] .= 1
    df_ = copy(df)[:, colnames]
    return 1 ./ (1 .+ exp.(-Matrix(df_[:, colnames]) * b))
end

function LogisticRegressionFilter!(df;
    coefs = Dict(
        "intercept"                 => -97.1879,
        "length_scale"              => 0.1267,
        "solidity"                  => 91.164,
        "b1_mean_reflectance"       => 7.354,
        "b7_mean_reflectance"       => -1.517,
        "b1_mean_boundary_contrast" => 2.239,
        )
    )
    colnames = [x for x in keys(coefs)]
    b = [x for x in values(coefs)]
    df[:, :intercept] = 1;
    df[:, :probability] = 1 ./ (1 .+ exp.(-Matrix(df[:, colnames]) * b))
end

"""
    add_mean_reflectance!(props_df, img, indices)

Compute the mean reflectance for `img` for each label in `props_df`. Assumes
that `props_df` contains labels corresponding to the dictionary `indices` 
(see @ref[`component_indices`]).
"""
function add_mean_reflectance!(props_df, img, indices)
    segment_mean_reflectance(r) = mean(img[indices[r]])
    props_df.mean_reflectance = segment_mean_reflectance.(props_df.label)
end

"""
    add_mean_boundary_reflectance!(props_df, img, labels; radius=15)

Compute the average of `img` within `radius` of the objects in `labels`. Uses
the bounding boxes in `props_df` so that they don't have to be re-computed.
"""
function add_mean_boundary_reflectance!(props_df, img, labels; radius=15)
    n, m = size(labels)
    bdry_ref = []
    for data in eachrow(props_df)
        # expand the bounding box by radius
        # minimum row is the maximum 
        rmin = maximum((data.min_row - radius, 1))
        rmax = minimum((data.max_row + radius, n))
        cmin = maximum((data.min_col - radius, 1))
        cmax = minimum((data.max_col + radius, m))

        label_subset = Int64.(labels[rmin:rmax, cmin:cmax] .== data.label)
        boundary = expand_labels(label_subset, radius)
        boundary[label_subset .> 0] .= 0
        image_subset = img[rmin:rmax, cmin:cmax]
        push!(bdry_ref, mean(image_subset[boundary .> 0]))
    end
    props_df.mean_boundary_reflectance = bdry_ref
end


### Helper for "missing" slots in the data retrieval
function fill_missing!(cases; template=Gray.(zeros(Bool, (400, 400))))
    for (idx, img) in enumerate(cases)
        if isnothing(img)
            cases[idx] = template
        end
    end
end

"""
Select floes with 


"""
function merge_floes(labeled_imgs, falsecolor_image, masks;
    max_distance_pixels=5,
    max_error_area=0.2,
    minimum_probability=0.5,
    )
    n = length(labeled_imgs)
    (n == 1) && return(labeled_imgs)

    # Initialize with the first image
    init_img = copy(labeled_imgs[1])
    init_indices = component_indices(init_img)
    for i in 2:n
        comp_img = copy(labeled_imgs[i])
        comp_indices = component_indices(comp_img)
        
        df1 = extended_regionprops(init_img)
        df2 = extended_regionprops(comp_img)

        df_comp = objectwise_compare_segmentation(df1, df2, labels1, labels2);

        # Merge criteria 1: Refining similar segments
        within_tolerance(d, e) = (d .< max_distance_pixels) .&& (e .< max_error_area)
        df_matches = subset(df_comp, [:dist_s1_s2, :scaled_relative_error_area] => within_tolerance)
        merge_arrays!(init_img, comp_img, df_matches; metric_variable=:probability)

        # Merge criteria 2: refining 
        # df_matches = subset(
    end
end

"""
    merge_arrays!(labels1, labels2, comparison_dataframe)

Remove labels
"""
function merge_arrays!(labels1, labels2, comparison_dataframe; metric_variable=:probability)    
    nrow(comparison_dataframe) > 0 && begin
        
        # Don't destroy the comparisons
        df_ = copy(comparison_dataframe)
        
        # Select the item in the relative set with lowest area difference.
        subset!(
            groupby(df_, :s1_label),
            :scaled_relative_error_area => r -> 1:length(r) .== argmin(r),
        )
        subset!(
            groupby(df_, :s2_label),
            :scaled_relative_error_area => r -> 1:length(r) .== argmin(r),
        )
    
        # Select the option with highest probability
        transform!(
            df_,
            [Symbol("s1_", metric_variable),
             Symbol("s2_", metric_variable)] =>
                ByRow((s1, s2) -> s1 .> s2) => :s1_better,
        )

        indices1 = component_indices(labels1)
        indices2 = component_indices(labels2)
        
        _remove_labels!(labels1, indices1, df_[.!df_.s1_better, :s1_label])
        _assign_labels!(labels1, indices2, df_[.!df_.s1_better, :s2_label]; 
            offset=maximum(labels1))
    end
end


function merge_floes(df1, df2, labels1, labels2; 
    max_distance_pixels=10,
    max_error_area=0.25,
    min_floe_size=100
    )

    # If no floes to merge, skip merge
    nrow(df1) == 0 && return labels2
    nrow(df2) == 0 && return labels1

    #### Set up starting images
    A = labels1
    B = labels2
    offset_b = maximum(A) # Offset the labels in B by the largest value in A
    A_indices = component_indices(A)
    B_indices = component_indices(B)
    A_labels = df1.label
    B_labels = df2.label

    F = zeros(Int64, size(A))

    #### Case 1: No overlap
    A_no_overlap = []
    B_no_overlap = []
    for L in A_labels
        if maximum(B[A_indices[L]]) == 0
            F[A_indices[L]] .= L
            push!(A_no_overlap, L)
        end
    end
    for L in B_labels
        if maximum(A[B_indices[L]]) == 0
            F[B_indices[L]] .= L + offset_b
            push!(B_no_overlap, L)
        end
    end

    subset!(df1, :label => ByRow(r -> r ∉ A_no_overlap))
    subset!(df2, :label => ByRow(r -> r ∉ B_no_overlap))
    nrow(df1) == 0 || nrow(df2) == 0 && return F

    #### Case 2: High-Quality Pairs
    # In this case, there exists at least one item in the relevant set where the error metrics are both within the tolerance.
    # Out of these objects, choose the one with the highest probability. 
    df_comp = objectwise_compare_segmentation(df1, df2, labels1, labels2);
    matches = subset(
        df_comp,
        [:dist_s1_s2, :scaled_relative_error_area] => (d, e) -> (d .< max_distance_pixels) .&& (e .< max_error_area),
    )
    nrow(matches) > 0 && begin
        # Select the item in the relative set with lowest area difference.
        subset!(
            groupby(matches, :s1_label),
            :scaled_relative_error_area => r -> 1:length(r) .== argmin(r),
        )
        subset!(
            groupby(matches, :s2_label),
            :scaled_relative_error_area => r -> 1:length(r) .== argmin(r),
        )

        # Select the option with highest probability
        transform!(
            matches,
            [:s1_probability, :s2_probability] =>
                ByRow((s1, s2) -> s1 .> s2) => :s1_better,
        )

        # Merge the two, prioritizing the second if there is overlap.
        A_labels = matches[matches.s1_better, :s1_label]
        B_labels = matches[.!matches.s1_better, :s2_label];

        for L in A_labels
            F[A_indices[L]] .= L
        end        
        for L in B_labels
            F[B_indices[L]] .= L + offset_b
        end

        # Add intersections to list
        idx = F .> 0
        A_labels = union(A_labels, unique(A[idx]))
        B_labels = union(B_labels, unique(B[idx]))

        # Update the dataframes to remove the resolved labels
        subset!(df1, :label => ByRow(r -> r ∉ A_labels))
        subset!(df2, :label => ByRow(r -> r ∉ B_labels))
    end

    #### Case 3: Poor matches, including over and undersegmentation
    # 1. Loop through remaining objects in A. If probability is higher
    #    for the object in A than all intersections in B, keep object.
    # 2. Loop through remaining objects in B. If no intersection with
    #    the objects kept in step 1, keep object.
    # 3. Update F and return.

    # Select objects in A with higher probability than any intersection with B
    A_labels = []
    B_probability = Dict(r => p for (r, p) in zip(df2.label, df2.probability))
    for s1 in eachrow(df1)        
        B_labels = filter(r -> r ∈ df2.label, unique(labels2[A_indices[s1.label]]))
        if all(s1.probability .> [B_probability[r] for r in B_labels])
            push!(A_labels, s1.label)
        end
    end
    for L in A_labels
        F[A_indices[L]] .= L
    end
    
    # Select objects in B with no intersection with F
    B_labels = unique(B[F .> 0])
    subset!(df2, :label => ByRow(r -> r ∉ B_labels))
    for L in df2.label
        F[B_indices[L]] .= L + offset_b
    end
    return F
end

import IceFloeTracker.Tracking: euclidean_distance

"""

Produce a dataframe linking objects between two labeled images if the overlap
between them is larger than 5% of either object.
"""
function compare_objects(
    df1, df2, labels1, labels2;
    indices1=component_indices(labels1),
    indices2=component_indices(labels2),
    comp_properties=propertynames(df1),
    tol_area_fraction=0.05,
)    


    no_overlaps1 = _nonoverlapping_labels(labels2, indices1, labels1)
    no_overlaps2 = _nonoverlapping_labels(labels1, indices2, labels2)
    overlaps1 = setdiff([l for l in df1.label], no_overlaps1)
    overlaps2 = setdiff([l for l in df2.label], no_overlaps2)
    
    forward_overlap = Dict(
        r => [filter(r -> r != 0, unique(labels2[indices1[r]]))]
        for r in overlaps_1)
    backward_overlap = Dict(
        r => [filter(r -> r != 0, unique(labels1[indices2[r]]))]
        for r in overlaps_2)

    # TODO: subset df_comp to just the ones with overlaps
    df_comp = copy(df1[:, comp_properties])
    rename!(df_comp,Dict(p => Symbol("s!_", p) for p in comp_properties))

    # TODO: determine if I need to do both directions, or if the overlaps are bidirectional

    # TODO: update the method below to use the set of overlaps not the relevant set
    
    properties = union(propertynames(df1), propertynames(df2))
    relevant_set = get_relevant_set(df1, df2, labels1, labels2)
    results = DataFrame[]
    for floe in eachrow(df1)
        g = floe.label
        g in keys(relevant_set) && begin
            df_rs = subset(df2, :label => ByRow(s -> s in relevant_set[g]))
            df_rs[:, :dist_s1_s2] = euclidean_distance(floe, df_rs; r=1) # r=1 means use pixel units, not meters
            df_rs[:, :scaled_relative_error_area] =
                abs.(df_rs.area .- floe.area) ./ (df_rs.area .+ floe.area)
            for colname in properties
                df_rs[!, Symbol("s1_", colname)] .= floe[colname]
            end
            push!(results, df_rs)
        end
    end
    if length(results) == 0
        return DataFrame(Dict(x=>[] for x in union(properties, [:s1_label, :s2_label, :dist_s1_s2, :scaled_relative_error_area])))
    end
    results_df = vcat(results...; cols=:union)
    rename!(results_df, Dict(r => Symbol("s2_", r) for r in properties))

    return results_df
end







function objectwise_compare_segmentation(
    df1, df2, labels1, labels2; extended=true
)    
    properties = union(propertynames(df1), propertynames(df2))
    relevant_set = get_relevant_set(df1, df2, labels1, labels2)
    results = DataFrame[]
    for floe in eachrow(df1)
        g = floe.label
        g in keys(relevant_set) && begin
            df_rs = subset(df2, :label => ByRow(s -> s in relevant_set[g]))
            df_rs[:, :dist_s1_s2] = euclidean_distance(floe, df_rs; r=1) # r=1 means use pixel units, not meters
            df_rs[:, :scaled_relative_error_area] =
                abs.(df_rs.area .- floe.area) ./ (df_rs.area .+ floe.area)
            df_rs[:, :relative_error_area] =
                abs.(df_rs.area .- floe.area) ./ floe.area

            # object-wise precision and recall
            gtmask = labels1 .== g
            pr = []
            re = []
            sd = []
            for s in df_rs.label
                smask = labels2 .== s
                intersect_area = sum(gtmask .&& smask)
                push!(pr, intersect_area / sum(smask))
                push!(re, intersect_area / sum(gtmask))
                push!(sd, sum(gtmask .|| smask) .- intersect_area)
            end
            df_rs[:, :precision] .= pr
            df_rs[:, :recall] .= re
            df_rs[:, :shape_difference] .= sd
            
            for colname in properties
                df_rs[!, Symbol("s1_", colname)] .= floe[colname]
            end
            push!(results, df_rs)
        end
        # else: add to no relevant set list
    end

    # for floe in eachrow(df2)
    # 
    if length(results) == 0
        return DataFrame(Dict(x=>[] for x in union(properties, [:s1_label, :s2_label, :dist_s1_s2, :scaled_relative_error_area])))
    end
    results_df = vcat(results...; cols=:union)
    rename!(results_df, Dict(r => Symbol("s2_", r) for r in properties))

    return results_df
end

"""
Helper functions for the merge_floes routine
"""
function _nonoverlapping_labels(other, indices, labels)
    return [
        label for label in labels
        if maximum(other[indices[label]]) == 0
    ]
end

function _assign_labels!(output, indices, labels; offset=0)
    foreach(labels) do label
        output[indices[label]] .= label + offset
    end
end

function _remove_labels!(output, indices, remove_labels)
    for L in remove_labels
        if L != 0
            output[indices[L]] .= 0
        end
    end
end


function colorize_classification(labeled_image;
color_map=Dict(
        0=>RGB(0),
        1=>RGB(0.018, 0.49, 0.64),
        2=>RGB(1),
        3=>RGB(0.84, 0.73, 0.94)
        )
    )
    return n0f8.(map(i -> color_map[i], labeled_image))
end


abstract type IceFloePreprocessingAlgorithm end
abstract type IceFloeClassificationAlgorithm end

"""
   Preprocess(
        adapthisteq_params = (nbins=256, rblocks=8, cblocks=8, clip=1)
    )
    Preprocess()(img, mask)

    Converts input image to grayscale, then preprocesses by applying contrast limited adaptive histogram
    equalization. The mask may include the land mask, coastal buffer, or a domain

"""
@kwdef struct Preprocess <: IceFloePreprocessingAlgorithm
    histogram_algorithm = ContrastLimitedAdaptiveHistogramEqualization
    histogram_params = (nbins=256, rblocks=4, cblocks=4, clip=1)
end

function (p::Preprocess)(
    image::AbstractArray{<:Union{AbstractGray, TransparentGray, AbstractRGB,TransparentRGB}}, landmask
)
    # Cast to grayscale first to save compute time
    proc_img = Gray.(image)
    apply_landmask!(proc_img, landmask)

    adjust_histogram!(
        proc_img,
        p.histogram_algorithm(;
            p.histogram_params...
        ),
    )

    # Re-apply mask so histogram adjustment doesn't bleed into land
    apply_landmask!(proc_img, landmask)
    return proc_img
end

"""
   Classify(
        τ₁=0.1,
        τ₂=0.2,
        τ₇=0.2,
        key=Dict("land"=>0, "water"=>1, "ice"=>2, "cloud"=>3)
    )
    Classify()(false_color_image, mask)

Classifies an image into land, water, ice, and cloud using the Watkins2026 cloud mask
and the IceDetectionBrightnessMidpoint algorithm. The parameter τ₁ is the brightness 
minimum for the ice detection algorithm, while the τ₂ and τ₇ parameters are used in the 
cloud mask algorithm. The `key` specifies the integers used to encode the classification
for the returned label map.

"""
@kwdef struct Classify <: IceFloeClassificationAlgorithm
        τ₁=0.1
        τ₂=0.2
        τ₇=0.2
        key=Dict("land"=>0, "water"=>1, "ice"=>2, "cloud"=>3)
end

function (c::IceFloeClassificationAlgorithm)(false_color_image, land_mask)::Matrix{Int64}
    cloud_mask_algorithm=Watkins2026CloudMask(band_2_threshold=c.τ₂, band_7_threshold=c.τ₇)
    ice_mask_algorithm=IceDetectionBrightnessMidpoint(; minimum_reflectance=c.τ₁)
    fc_masked = apply_landmask(false_color_image, land_mask)
    clouds = cloud_mask_algorithm(fc_masked)
    ice = Gray.(blue.(apply_landmask(fc_masked, clouds))) |> ice_mask_algorithm
    
    classified_image = ones(Int64, size(false_color_image)) .* c.key["water"]
    classified_image[land_mask .> 0] .= c.key["land"]
    classified_image[ice .> 0] .= c.key["ice"]
    classified_image[clouds .> 0] .= c.key["cloud"]
    return classified_image
end

function colorize_classification(labeled_image;
    color_map=Dict(
        0=>RGB(0),
        1=>RGB(0.018, 0.49, 0.64),
        2=>RGB(1),
        3=>RGB(0.84, 0.73, 0.94)
        )
    )
    return n0f8.(map(i -> color_map[i], labeled_image))
end

# TODO: refine segmentation boundaries using the boundary splines, and use the distance function 
# to settle differences. 
