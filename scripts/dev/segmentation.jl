#### Functions in the FSPipeline, placed here for early access ####

function filter_floes(
    img_indexmap,
    coastal_buffer_mask,
    cloud_mask,
    falsecolor_image;
    min_floe_size=100,
    max_floe_size=90_000,
    boundary_radius=15,
    min_reflectance=0.4,
    min_circularity=0.3,
    min_solidity=0.7,
    min_contrast=0.01,
    filter_function=LogisticRegressionFilter,
    min_probability=0.5,
)
    # 1. Remove objects which overlap the coastal mask
    overlap = unique(img_indexmap[coastal_buffer_mask])
    indices = component_indices(img_indexmap)
    for L in overlap
        img_indexmap[indices[L]] .= 0
    end

    # 2. Remove objects outside the specified size bounds prior to extracting features.
    # This is important since the small features can cause problems in some feature
    # descriptors.
    remove_small_segments!(img_indexmap, min_floe_size)
    remove_large_segments!(img_indexmap, max_floe_size)

    # 3. Get object-wise properties
    results_df = regionprops_table(img_indexmap;
        properties=[:label, :area, :perimeter, :bbox, :centroid, :convex_area,
                    :major_axis_length, :minor_axis_length, :orientation],
        convex_area_algorithm=PolygonConvexArea()
    )
    # Return blank image if no floes remain
    nrow(results_df) == 0 && return results_df

    results_df[:, :length_scale] = results_df[:, :area] .^ 0.5
    results_df[:, :circularity] = 4 * π * results_df[:, :area] ./ results_df[:, :perimeter] .^ 2
    subset!(results_df, :circularity => r -> r .> min_circularity)
    results_df[:, :solidity] = results_df[:, :area] ./ results_df[:, :convex_area]
    subset!(results_df, :solidity => r -> r .> min_solidity)
    nrow(results_df) == 0 && return results_df

    results_df[:, :cloud_fraction] =  (r -> mean(cloud_mask[indices[r]])).(results_df[:, :label])
    
    # mean reflectance
    segment_mean_reflectance = segment_mean(SegmentedImage(falsecolor_image, img_indexmap))
    b = [segment_mean_reflectance[L] for L in  results_df[:, :label]]
    results_df[:, :b1_reflectance_mean] = blue.(b)
    results_df[:, :b7_reflectance_mean] = red.(b)
    results_df[:, :b2_reflectance_mean] = green.(b)
    
    subset!(results_df, :b1_reflectance_mean => r -> r .> min_reflectance)
    nrow(results_df) == 0 && return results_df
    
    # mean boundary reflectance
    b1 = blue.(falsecolor_image)
    bdry_indexmap = expand_labels(img_indexmap, boundary_radius) .- img_indexmap
    bdry_indices = component_indices(bdry_indexmap)
    bdry_labels = intersect(results_df[:, :label], unique(bdry_indexmap))
    b1_bdry_means = Dict(L => mean(b1[bdry_indices[L]]) for L in bdry_labels)
    for L ∈ results_df[:, :label]
        if L ∉ bdry_labels
            push!(b1_bdry_means, L => 0)
        end
    end
    results_df[:, :b1_reflectance_bdry_mean] = [b1_bdry_means[L] for L in results_df[:, :label]]
    results_df[:, :b1_bdry_contrast] = results_df[:, :b1_reflectance_mean] .- results_df[:, :b1_reflectance_bdry_mean]
    subset!(results_df, :b1_bdry_contrast => r -> r .> min_contrast)
    nrow(results_df) == 0 && return results_df

    results_df[:, :probability] .= filter_function(results_df)
    subset!(results_df, :probability => r -> r .> min_probability)

    return results_df
end

function LogisticRegressionFilter(df;
    coefs = Dict(
        "intercept"           => -97.1879,
        "length_scale"        => 0.1267,
        "solidity"            => 91.164,
        "b1_reflectance_mean" => 7.354,
        "b1_bdry_contrast"    => 2.239,
        "b7_reflectance_mean" => -1.517,
        )
    )
    colnames = [x for x in keys(coefs)]
    b = [x for x in values(coefs)]
    df[:, :intercept] .= 1;
    return 1 ./ (1 .+ exp.(-Matrix(df[:, colnames]) * b))
end

### Helper for "missing" slots in the data retrieval
function fill_missing!(cases; template=Gray.(zeros(Bool, (400, 400))))
    for (idx, img) in enumerate(cases)
        if isnothing(img)
            cases[idx] = template
        end
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

function objectwise_compare_segmentation(
    df1, df2, labels1, labels2
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

function get_relevant_set(df1, df2, labels1, labels2)
    relevant_set = Dict{Int64,Vector{Int64}}()
    for floe in eachrow(df1)
        # select labels that are inside the bounding box for the floe
        matched_labels = unique(
            labels2[floe.min_row:floe.max_row, floe.min_col:floe.max_col]
        )

        # if any, then check centroid positions
        maximum(matched_labels) == 0 && continue
        # get the rows in the segments_df from the matched labels
        candidate_subset = subset(df2, :label => ByRow(l -> l in matched_labels))

        relevant_set_labels = []

        # check if centroid g in s
        rc = round(Int64, floe.row_centroid)
        cc = round(Int64, floe.col_centroid)
        push!(relevant_set_labels, labels2[rc, cc])

        # check if centroid s in g
        for s_floe in eachrow(candidate_subset)
            rc = round(Int64, s_floe.row_centroid)
            cc = round(Int64, s_floe.col_centroid)
            (labels1[rc, cc] == floe.label) && begin
                push!(relevant_set_labels, s_floe.label)
            end

            # joint bbox
            rmin = minimum((floe.min_row, s_floe.min_row))
            rmax = maximum((floe.max_row, s_floe.max_row))
            cmin = minimum((floe.min_col, s_floe.min_col))
            cmax = maximum((floe.max_col, s_floe.max_col))

            # check if area overlap between g and s is larger than 50% of g
            gtmask = labels1[rmin:rmax, cmin:cmax] .== floe.label
            slmask = labels2[rmin:rmax, cmin:cmax] .== s_floe.label
            intersect_area = sum(gtmask .&& slmask)
            if maximum([intersect_area / s_floe.area, intersect_area / floe.area]) > 0.5
                push!(relevant_set_labels, s_floe.label)
            end
        end
        relevant_set_labels = filter(r -> r != 0, unique(relevant_set_labels))
        if length(relevant_set_labels) > 0
            push!(relevant_set, floe.label => relevant_set_labels)
        end
    
    end
    return relevant_set
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
    end
    if length(results) == 0
        return DataFrame(Dict(x=>[] for x in union(properties, [:s1_label, :s2_label, :dist_s1_s2, :scaled_relative_error_area])))
    end
    results_df = vcat(results...; cols=:union)
    rename!(results_df, Dict(r => Symbol("s2_", r) for r in properties))

    return results_df
end

