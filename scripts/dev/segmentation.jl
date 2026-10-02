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
    # If this is outside the function we can do direct tests
    bdry_mean(row) = _get_boundary_mean(row, img, labels, radius)
    props_df.mean_boundary_reflectance = bdry_mean.(eachrow(props_df))
end

function _get_boundary_mean(dataframe_row, img, labels, radius)
    # expand the bounding box by radius
    # minimum row is the maximum 
    n, m = size(labels)
    rmin = maximum((dataframe_row.min_row - radius, 1))
    rmax = minimum((dataframe_row.max_row + radius, n))
    cmin = maximum((dataframe_row.min_col - radius, 1))
    cmax = minimum((dataframe_row.max_col + radius, m))

    label_subset = Int64.(labels[rmin:rmax, cmin:cmax] .== dataframe_row.label)
    boundary = expand_labels(label_subset, radius)
    boundary[label_subset .> 0] .= 0
    image_subset = img[rmin:rmax, cmin:cmax]
    return mean(image_subset[boundary .> 0])
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
    compare_objects(
        df1, df2, labels1, labels2;
        indices1=component_indices(labels1),
        indices2=component_indices(labels2),
        comp_properties=[
            :label, :area, :row_centroid, :col_centroid,
            :max_col, :max_row, :min_col, :min_row, :probability
        ],
        tol_area_fraction=0.05,
    )

Produce a dataframe comparing objects in a pair of labeled images, including 
all paired labels between labels1 and labels2 with area overlap greater than
`tol_area_fraction` relative to either label. Additionally computes the distance between centroids, area overlap, and fractional area overlap.

Inputs:
    - `df1` = region properties dataframe from labels1
    - `df2` = region properties dataframe from labels2
    - `labels1` = labeled image (Matrix{Int64})
    - `labels2` = labeled image (Matrix{Int64})
    - `indices1=component_indices(labels1)` = Indices map, option to reuse from earlier in processing 
    - `indices2=component_indices(labels2)` = Indices map, option to reuse from earlier
    - `comp_properties=[
            :label, :area, :row_centroid, :col_centroid,
            :max_col, :max_row, :min_col, :min_row, :probability
        ]` = Columns in df1 and df2 to include in comparison
    - `tol_area_fraction=0.05`= Minimum area fraction to include in comparison
"""
function compare_objects(
    df1::DataFrame,
    df2::DataFrame, 
    labels1::Matrix{Int64},
    labels2::Matrix{Int64}; # Should this be keyword or no?
    indices1=component_indices(labels1),
    indices2=component_indices(labels2),
    comp_properties=[
        :label, :area, :row_centroid, :col_centroid,
        :max_col, :max_row, :min_col, :min_row, :probability
    ],
    tol_area_fraction=0.05, # TODO: decide whether we should filter probability here
)::DataFrame

    # Get list of labels in 1 with nonzero intersection
    no_overlaps = _nonoverlapping_labels(labels2, indices1, df1.label)
    overlaps = setdiff(df1.label, no_overlaps)

    # Make list of intersections from 1 to 2
    s1_label_list = []
    s2_label_list = []
    for r in overlaps
        for s in filter(r -> r != 0, unique(labels2[indices1[r]]))
            append!(s1_label_list, r)
            append!(s2_label_list, s)
        end
    end

    # Generate joint dataframe
    df_comp1 = rename(df1[:, comp_properties],
        Dict(p => Symbol("s1_", p) for p in comp_properties))
    df_comp2 = rename(df2[:, comp_properties],
        Dict(p => Symbol("s2_", p) for p in comp_properties))
    df_dict1 = Dict(row.s1_label => row for row in eachrow(df_comp1))
    df_dict2 = Dict(row.s2_label => row for row in eachrow(df_comp2))
    df_comp = hcat(
        DataFrame([df_dict1[l] for l in s1_label_list]), 
        DataFrame([df_dict2[l] for l in s2_label_list])
    )

    # Compute overlap metrics
    transform!(df_comp,
        [:s1_row_centroid, :s2_row_centroid,
         :s1_col_centroid, :s2_col_centroid] => 
        ByRow((r1, r2, c1, c2) -> sqrt((r1 - r2)^2 + (c1 - c2)^2)) =>
        :s1_s2_dist
    )

    transform!(df_comp, 
        [:s1_label, :s2_label, 
         :s1_min_row, :s1_max_row, :s1_min_col, :s2_max_col] =>
        ByRow((l1, l2, rmin, rmax, cmin, cmax) ->
            sum(
                (labels1[rmin:rmax, cmin:cmax] .== l1) .&&
                (labels2[rmin:rmax, cmin:cmax] .== l2)
                )
            ) =>
        :s1_s2_area_overlap
    )

    transform!(df_comp,
        [:s1_s2_area_overlap, :s1_area] => ByRow((a0, a1) -> a0/a1) =>
        :s1_area_fraction
    )

    transform!(df_comp,
        [:s1_s2_area_overlap, :s2_area] => ByRow((a0, a1) -> a0/a1) =>
        :s2_area_fraction
    )

    subset!(df_comp, :s1_area_fraction => r -> r .> tol_area_fraction)
    subset!(df_comp, :s2_area_fraction => r -> r .> tol_area_fraction)
    
    return df_comp
end


"""
    _nonoverlapping_labels(other, indices, labels)

Return a list of labels in matrix `other` which have no
overlap with the list of labels `labels` and the corresponding
indices dictionary `indices`. Both `labels` and `indices` 
come from a second labeled indexmap to be compared with `other`.

"""
function _nonoverlapping_labels(other, indices, labels)
    return [
        label for label in labels
        if maximum(other[indices[label]]) == 0
    ]
end

"""
    _assign_labels!(output, indices, labels; offset=0)

Insert each label from list `labels` into `output` using 
the indices dictionary `indices`. Optional `offset` integer
can be added to avoid duplicating an existing label.

"""
function _assign_labels!(output, indices, labels; offset=0)
    foreach(labels) do label
        output[indices[label]] .= label + offset
    end
end

"""
    _remove_labels!(output, indices, remove_labels)

Remove regions of `output` by setting the indices to 0.
The labels in `remove_labels` correspond to the dictionary
keys in `indices`.

"""
function _remove_labels!(output, indices, remove_labels)
    for L in remove_labels
        if L != 0
            output[indices[L]] .= 0
        end
    end
end


"""
    sequential_merge_floes(labeled_imgs, falsecolor_image, masks;
        comp_properties=[
            :label, :area, :row_centroid, :col_centroid,
            :max_col, :max_row, :min_col, :min_row, :probability
        ],
        tol_area_fraction=0.05,
    )

Sequentially compare the images in `labeled_imgs` using the `compare_objects` 
function. Use the area average of floe probabilities to compare - winner take all.
(e.g., if S1 intersects T1 and T2, then we keep S1 if its probability is higher than
the area-weighted average probability of T1 and T2). Returns a single labeled image.

"""
function sequential_merge_floes(labeled_imgs, falsecolor_image, masks;
    comp_properties=[
        :label, :area, :row_centroid, :col_centroid,
        :max_col, :max_row, :min_col, :min_row, :probability
    ],
    tol_area_fraction=0.05,
    )
    n = length(labeled_imgs)
    (n == 1) && return(labeled_imgs)

    # Initialize with the first image
    init_img = copy(labeled_imgs[1])
    init_indices = component_indices(init_img)

    # TODO: Could speed up by getting minimal set of properties
    df1 = extended_regionprops_table(
        init_img, falsecolor_image, masks
    )
    
    for i in 2:n
        comp_img = copy(labeled_imgs[i])
        comp_indices = component_indices(comp_img)
        
        df2 = extended_regionprops_table(
            comp_img, falsecolor_image, masks
        )

        df_comp = compare_objects(
            df1, df2,
            init_img, comp_img;
            indices1=init_indices, 
            indices2=comp_indices,
            comp_properties=comp_properties,
            tol_area_fraction=tol_area_fraction
        )

        # Method 1: Compare with full set of intersections
        transform!(
            groupby(df_comp, :s1_label),
            [:s2_area, :s2_probability] =>
            ((a, p) -> sum(p .* a ./ sum(a))) =>
            :s2_weighted_probability
        )
        df_sel = subset(
            df_comp, [:s1_probability, :s2_weighted_probability] => 
            (p1, p2) -> p1 .< p2
        )

        remove_labels = df_sel.s1_label
        no_matches = setdiff(df_comp.s2_label, df2.label)
        add_labels = union(df_sel.s2_label, no_matches)

        if (length(remove_labels) > 0) || (length(add_labels) > 0)
            merge_arrays!(
                init_img, init_indices, comp_indices,
                remove_labels, add_labels
            )
            
            # update information for init_img
            # Could be a clever way to join df1 and df2
            # instead of recomputing
            df1 = extended_regionprops_table(
                init_img, falsecolor_image, masks
            )
            init_indices = component_indices(init_img)
        end
    end
    return init_img
end

"""
    merge_arrays!(
        output,
        indices1,
        indices2,
        remove_labels,
        add_labels
    )

Update labels1 by (1) removing the labels for each `L1` the list `remove_labels` by setting everything in `indices1[L1]` to 0 and then (2)
writing `L2` into labels1 for each `L2` in `add_labels`.
"""

function merge_arrays!(output, indices1, indices2, remove_labels, add_labels)
    _remove_labels!(output, indices1, remove_labels)
    _assign_labels!(output, indices2, add_labels;
        offset=maximum(labels1))
end


"""
    colorize_classification(labeled_image;
                            color_map=Dict(
                                    0=>RGB(0),
                                    1=>RGB(0.018, 0.49, 0.64),
                                    2=>RGB(1),
                                    3=>RGB(0.84, 0.73, 0.94)
                                    )
                                )

Colorize a labeled image by mapping the keys in `color_map` to 
entry colors. `color_map` needs to include all the labels in
`labeled_image`. By default, the colors correspond to black, blue,
white, and purple.

"""
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
