using Pkg
Pkg.activate("calval")
Pkg.instantiate()
using IceFloeTracker
using Images
using Dates

dataset = Watkins2026Dataset(ref="main")
fc_imgs = modis_falsecolor.(dataset)
land_masks = modis_landmask.(dataset) .|> r -> Gray.(r) .> 0

"""

Produces a segmented image with up to 4 categories: land, water, ice, and cloud.

"""
function ift_classification(false_color_image, land_mask;
        tau_1=0.1,
        tau_2=0.2,
        tau_7=0.2,
        label_map=Dict("land"=>0, "water"=>1, "ice"=>2, "cloud"=>3)
    )
    
    cloud_mask_algorithm=Watkins2026CloudMask(
        band_7_threshold = tau_7,
        band_2_threshold = tau_2,
        opening_strel = strel_disk(3),
        dilation_strel = strel_disk(2),
        min_hole_size = 300,
        max_fill_size = 1e4,
        min_contrast = 0.2,
        ) 

    ice_mask_algorithm=IceDetectionBrightnessMidpoint(minimum_reflectance=tau_1)
    
    coastal_buffer = create_coastal_buffer_mask(land_mask .> 0, strel_disk(25))
    fc_masked = apply_landmask(false_color_image, coastal_buffer)
    clouds = cloud_mask_algorithm(fc_masked) .> 0
    band_1_masked = Gray.(blue.(apply_landmask(fc_masked, clouds)))
    ice = ice_mask_algorithm(band_1_masked) .> 0
    
    classified_image = ones(Int64, size(false_color_image)) .* label_map["water"]
    classified_image[coastal_buffer] .= label_map["land"]
    classified_image[ice] .= label_map["ice"]
    classified_image[clouds] .= label_map["cloud"]
    
    return SegmentedImage(false_color_image, classified_image)
end

classified = ift_classification.(fc_imgs, land_masks)
images = view_seg_random.(classified)
data = labels_map.(classified)

file_names = [
     join(
        [lpad(cn, 3, "0"), r, Dates.format(d, "yyyymmdd"), s, "binary_water_samples.png"],
        "-"
    )
    for (cn, r, s, d) in zip(
            dataset.info.case_number,
            dataset.info.region,
            dataset.info.satellite,
            dataset.info.start_date
            )
    ]
for (idx, data) in enumerate(eachrow(dataset.info))
    fname = join(
        [
            lpad(data.case_number, 3, "0"),
            data.region,
            Dates.format(data.start_date, "yyyymmdd"),
            data.satellite,
            "classified.png"
        ], "-")
    save(joinpath("../data/classification_results/images", fname), images[idx])
    save(joinpath("../data/classification_results", replace(fname, "png"=>"tiff")), Gray.(data[idx]./4))
end