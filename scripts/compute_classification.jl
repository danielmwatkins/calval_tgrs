using Pkg
Pkg.activate("calval")
Pkg.instantiate()
using IceFloeTracker
using Images
using Dates

dataset = Watkins2026Dataset(ref="main")
fc_imgs = modis_falsecolor.(dataset)
land_masks = modis_landmask.(dataset) .|> r -> Gray.(r) .> 0

include("dev/segmentation.jl")

dataloc = "/Users/dwatkin2/Documents/research/data/ice-floe-validation-dataset/500km/modis_500km/"



classified = ift_classification.(fc_imgs, land_masks)
images = _colorize_classification.(classified)

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
end