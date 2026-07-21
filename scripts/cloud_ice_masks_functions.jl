function CloudMask(fc_img;
        tau_b7=0.2,
        tau_b2=0.3,
        opening_strel=strel_disk(3),
        dilation_strel=strel_disk(2),
        min_hole_size=300,
        min_contrast=0.2,
        max_fill_size=10_000
    )
    
    b2 = green.(fc_img)
    b7 = red.(fc_img)
    init_mask = (b2 .> tau_b2) .&& (b7 .> tau_b7)
    
    # end early if no pixels flagged
    !maximum(init_mask) && return init_mask

    # remove speckle
    markers = opening(init_mask, opening_strel)
    init_mask .= mreconstruct(dilate, markers, init_mask, strel_diamond((3, 3)))
    
    # expand mask
    init_mask .= dilate(init_mask, dilation_strel)

    # label blank components for object-based methods (background == clouds)
    labels = label_components(.!init_mask) 
    remove_small_segments!(labels, min_hole_size)
    maximum(labels) .== 0 && return labels .== 0

    # remove low-contrast segmentsf
    remove_low_contrast_segments!(labels, b2, min_contrast, max_fill_size)

    return labels .== 0
end

function remove_small_segments!(labels, min_size)
    areas = component_lengths(labels)
    indices = component_indices(labels)

    for L in keys(areas)
        (L != 0) && begin
            (areas[L] < min_size) && begin
                labels[indices[L]] .= 0
            end
        end
    end
end

"""

Computes the contrast (min-max range) of each segment based on ref_img and sets
those with contrast less than `min_contrast` to zero. Only removes segments which
are smaller than max fill size.

"""
function remove_low_contrast_segments!(labels, ref_img, min_contrast, max_fill_size)
    areas = component_lengths(labels)
    indices = component_indices(labels)
    for L in unique(labels)
        (L != 0) && (areas[L] < max_fill_size) && begin
            contrast = maximum(ref_img[indices[L]]) - minimum(ref_img[indices[L]])
            (contrast < min_contrast) && begin
                labels[indices[L]] .= 0
            end
        end
    end
end

function component_contrast(labels, ref_img)
    indices = component_indices(labels)
    contrast = Dict(
        L => maximum(ref_img[indices[L]]) - minimum(ref_img[indices[L]]) for L in unique(labels)
    )
    return contrast
end

land_color = RGB(0)
ocean_color = RGB(46/255, 124/255, 163/255)
cloud_color = RGB(216/255, 182/255, 240/255)
floe_color = RGB(1)
floe_color_cloud = RGB(237/255, 154/255, 10/255)

function visualize_positive_cloud(cloud_mask, land_mask; land_color=land_color, ocean_color=ocean_color, cloud_color=cloud_color)
    img = RGB.(ocean_color .* ones(size(cloud_mask))) 
    img[land_mask .> 0] .= land_color
    img[(cloud_mask .> 0) .&& (land_mask .== 0)] .= cloud_color
    return img
end

function visualize_ice_under_cloud(floe_mask, cloud_mask, land_mask;
        land_color=land_color, ocean_color=ocean_color, cloud_color=cloud_color,
        floe_color=floe_color, floe_color_cloud=floe_color_cloud)
    img = RGB.(ocean_color .* ones(size(cloud_mask))) 
    img[land_mask .> 0] .= land_color
    img[(cloud_mask .> 0) .&& (land_mask .== 0)] .= cloud_color
    img[(floe_mask .> 0)] .= floe_color
    img[(cloud_mask .&& floe_mask .> 0)] .= floe_color_cloud
    return img
end

function visualize_cloud_confusion_matrix(julia_cloud_mask, modis_cloud_mask, land_mask;
        tpcolor=RGB(160/255, 191/255, 242/255),
        fpcolor=RGB(240/255, 198/255, 139/255),
        tncolor=RGB(119/255, 50/255, 153/255),
        fncolor=RGB(247/255, 101/255, 128/255),
        land_color=RGB(0)
    )
    
    ift_mask = julia_cloud_mask .> 0
    
    TP = ift_mask .&& modis_cloud_mask
    FP = ift_mask .&& .! modis_cloud_mask
    TN = .! ift_mask .&& .! modis_cloud_mask
    FN = .! ift_mask .&& modis_cloud_mask
    
    img = RGB.(land_mask) 
    img[TP] .= tpcolor
    img[FP] .= fpcolor
    img[TN] .= tncolor
    img[FN] .= fncolor
    img[land_mask .> 0] .= land_color
    
    return img
end

function confusion_matrix(ground_truth, predicted, mask)
    tp = sum(ground_truth .&& predicted .&& .! mask)
    fp = sum(.!ground_truth .&& predicted .&& .! mask)
    tn = sum(.!ground_truth .&& .!predicted .&& .! mask)
    fn = sum(ground_truth .&& .!predicted .&& .! mask)
    n = sum( .! mask)
    return DataFrame(tp=tp, fp=fp, tn=tn, fn=fn, n=n)
end
