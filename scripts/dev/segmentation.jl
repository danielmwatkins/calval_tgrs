"""Stitches clusters across tile boundaries based on neighbor with largest shared boundary.
The algorithm finds all pairs of segment labels at the tile edges. Then, we count the number of 
times each right-hand label is paired to a left-hand label, and for pairs with at least 4 pixel overlap,
the right-hand label is assigned as a candidate pair to the left-hand label. If the difference in grayscale
intensity is less than 0.1, the objects are merged. The function returns an image index map.
"""

function stitch_clusters(tiles, segmented_image, minimum_overlap=4, grayscale_threshold=0.1) 
    grayscale_magnitude(c) = Float64(Gray(c))
    
    idxmap = deepcopy(segmented_image.image_indexmap)
    n, m = size(idxmap)
    
    # Get the columns and rows used in the tiling algorithm
    interior_rows = []
    interior_cols = []
    
    for tile in tiles
        nrange, mrange = tile
        tn = maximum(nrange)
        tm = maximum(mrange)
        label_pairs = []
        if tn != n
            push!(label_pairs, vec([(x, y) for (x, y) in zip(idxmap[tn, :], idxmap[tn .+ 1, :])]))
        end
        
        if tm != m
            push!(label_pairs, vec([(x, y) for (x, y) in zip(idxmap[:, tm], idxmap[:, tm .+ 1])]))
        end

        if !isempty(label_pairs)

            # create a dataframe with the results
            label_pairs = vcat(label_pairs...)
            label_pairs = reshape(reinterpret(Int64, label_pairs), (2,:))
            df = DataFrame(left=label_pairs[1,:], right=label_pairs[2,:])
            
            # groupby right -> left pairs and get counts
            df_counts = combine(groupby(df, [:right, :left]), nrow => :count)
            
            # only use pairs that overlap by at least 2 pixels.
            df_counts = df_counts[df_counts.count .>= minimum_overlap, :]

            # only use pairs where left is not equal to right
            df_counts = df_counts[df_counts.right .!= df_counts.left, :]

            # don't merge if the segments are too different in color
            left_brightness = [grayscale_magnitude(segment_mean(segmented_image, l)) for l in df_counts.left]
            right_brightness = [grayscale_magnitude(segment_mean(segmented_image, r)) for r in df_counts.right]
            diff_means = abs.(right_brightness .- left_brightness)
            df_counts = df_counts[diff_means .< grayscale_threshold, :]
            
            if !isempty(df_counts)
                # now find the maximum overlapping segment for each
                df_pairs = combine(sdf -> sdf[argmax(sdf.count), [:right, :left, :count]], groupby(df_counts, :right))
            
                # make a lookup table and lookup function
                lut = Dict(ri => li for (ri, li) in zip(df_pairs.right, df_pairs.left))
                
                lookup_remap(ii) = begin
                    if haskey(lut, ii)
                       return lut[ii]
                    end
                    return ii
                end
    
                idxmap .= map(i -> lookup_remap(i), idxmap)
            end
        end    
    end
    return idxmap
end