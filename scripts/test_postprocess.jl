using Pkg
Pkg.activate("cal-val")
Pkg.add(;name="IceFloeTracker", rev="main") 
using Images
using IceFloeTracker
using DataFrames
using Random

#### Helper functions ####
# Produce an image by replacing values inside a segment with the segment mean color
function view_seg(s)
    map(i->segment_mean(s,i), labels_map(s))
end
# Return a random color of type RGB{N0f8}
function get_random_color(seed)
    Random.seed!(seed)
    rand(RGB{N0f8})
end

# Assign random colors to each segment (useful if viewing cluster results)
function view_seg_random(s)
    map(i->get_random_color(i), labels_map(s))
end

function bwdist(bwimg::AbstractArray{Bool})::AbstractArray{Float64}
    return distance_transform(feature_transform(bwimg))
end

function set_labels_to_background(seg, label_list)
    idx_map = seg.image_indexmap
    img = view_seg(seg)
    for s in label_list
        idx_map[idx_map .== s] .= 0
    end
    return SegmentedImage(img, idx_map)
end

function remove_small_floes(seg, min_floe_area)
    new_indexmap = deepcopy(seg.image_indexmap)
    img = view_seg(seg)
    small_floes = [s for s in segment_labels(seg) if segment_pixel_count(seg, s) < min_floe_area]
    return set_labels_to_background(seg, small_floes)
end

function get_subset_by_labels(s, label_list, img)
    idxmap = deepcopy(s.image_indexmap)
    for label in segment_labels(s)
        if label ∉ label_list
            idxmap[idxmap .== label] .= 0
        end
    end
    return SegmentedImage(img, idxmap)
end

function separate_by_watershed(seg;
        erode_se=strel_box((3,3)), dist_thresh=3)
    
    new_indexmap = deepcopy(seg.image_indexmap)
    img = view_seg(seg)
    bw = new_indexmap .> 0
    markers = erode(bw, strel_box((3,3)))
    d = bwdist(.!markers)
    cc = label_components(d .> dist_thresh)
    w = Images.watershed(bw, cc)
    lmap = labels_map(w)
    borders = isboundary(lmap) .> 0
    new_indexmap[isboundary(lmap) .> 0] .= 0

    new_indexmap .= label_components(new_indexmap)
    return SegmentedImage(img, new_indexmap)
end

function postprocess_floes(segments, img, landmask_img;
        landmask_se=strel_box((25,25)),
        min_area=64,
        solidity_threshold=0.8,
        watershed_erode_se=strel_diamond((3,3)),
        watershed_dist_thresh=3)

    processed = deepcopy(segments)
    properties = ["label", "area", "bbox", "centroid", 
    "convex_area", "major_axis_length", "minor_axis_length", "orientation", "perimeter"]
    
    @info "Remove floes that intersect dilated landmask"
    landmask = create_landmask(landmask_img, strel_box((25,25)));
    intersect_landmask = unique(processed.image_indexmap .* landmask.dilated)
    processed = set_labels_to_background(processed, intersect_landmask)

    @info "Watershed separation of low-solidity objects"
    # Set aside objects with high solidity. Then use the watershed transform to attempt to separate the low-solidity objects
    props = regionprops_table(processed.image_indexmap; properties=properties)
    props[:, "solidity"] = props.area ./ props.convex_area;
    high_solidity_labels = subset(props[:, [:solidity, :label]], [:solidity] => ByRow(r -> r >= solidity_threshold))[:, :label];
    low_solidity_labels = subset(props[:, [:solidity, :label]], [:solidity] => ByRow(r -> r < solidity_threshold))[:, :label];

    high_solidity = get_subset_by_labels(processed, high_solidity_labels, img)
    low_solidity = get_subset_by_labels(processed, low_solidity_labels,  img)
    low_solidity = separate_by_watershed(low_solidity; erode_se=watershed_erode_se, dist_thresh=watershed_dist_thresh)

    props = regionprops_table(low_solidity.image_indexmap; properties=properties)
    props[:, "solidity"] = props.area ./ props.convex_area;
    high_solidity_labels = subset(props[:, [:solidity, :label]], [:solidity] => ByRow(r -> r >= 0.85))[:, :label];
    watershed_high_solidity = get_subset_by_labels(low_solidity, high_solidity_labels, img)

    # combine for final segmented image
    final_labels = label_components(high_solidity.image_indexmap)
    watershed_labels = label_components(watershed_high_solidity.image_indexmap)
    watershed_labels[watershed_labels .> 0] .+=  maximum(final_labels)
    
    final_labels .= final_labels .+ watershed_labels
    final_segments = SegmentedImage(img, final_labels)

    @info "Remove small floes"
    final_segments = remove_small_floes(final_segments, min_area)
    
    return final_segments
end

# Initialize!
p = LopezAcosta2019.Segment(landmask_structuring_element=strel_box((11,11)));

# Large images
dataloc = "../../data/modis_data/fram_strait_2014/"
lm_img = float64.(RGB.(load(dataloc*"/landmask.tiff")))

for date in ["20140426", "20140427", "20140428", "20140429", "20140430", "20140501", "20140502", "20140503"]
    for satellite in ["aqua", "terra"]
        @info "Beginning "*date*" "*satellite
        
        tc_img = float64.(RGB.(load(dataloc*"truecolor/"*date*"."*satellite*".truecolor.250m.tiff")))
        fc_img = float64.(RGB.(load(dataloc*"falsecolor/"*date*"."*satellite*".falsecolor.250m.tiff")))
        
        @time begin
            init_segmented = p(tc_img, fc_img, lm_img)
        end
        
        @time begin
            postprocess_segmented = postprocess_floes(init_segmented, fc_img, lm_img)
        end
        
        save(dataloc*"labeled_init/"*date*"."*satellite*".colorized_init.tiff", view_seg_random(init_segmented) .* (init_segmented.image_indexmap .> 0))
        save(dataloc*"labeled_proc/"*date*"."*satellite*".colorized_proc.tiff", view_seg_random(postprocess_segmented) .* (postprocess_segmented.image_indexmap .> 0))
    end
end
