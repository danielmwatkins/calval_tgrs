using Pkg
Pkg.activate("../scripts/calval")

using CairoMakie
using CSV
using DataFrames
using Dates
using IceFloeTracker
using Images
using StatsBase

# We'll use the Classify functor and the colorize_classification function
include("../scripts/dev/segmentation.jl")

# Load 500km sample data
data = DataFrame(CSV.File("../data/calval_data_500km_samples.csv"))
# Scale inputs to between 0 and 1
for col in [:b7_mean, :b2_mean, :b1_mean]
    data[:, col] .= data[:, col] ./ 255
end

# Add information to dataframe, format case names, initialize classification column
data[:, :parsed_date] = Date.(string.(data[:, :date]), "yyyymmdd")
data[:, :month] = month.(data.parsed_date);
data[:, :case_number_string] = (c -> lpad(string(c), 3, "0")).(data.case_number);
data[:, :case] = [join([c, r, "500km", d], "-") for (c, r, d) in zip(data.case_number_string, data.region, data.date)];
data[:, :ift_classification] .= -1;

# Classifier with the new default settings
classifier = Classify(
    bright_ice_band_1_minimum=0.681,
    cloud_band_2_minimum=0.147,
    cloud_band_7_minimum=0.254,
    water_band_1_maximum=0.106
)

dataloc = "/Users/dwatkin2/Documents/research/data/ice-floe-validation-dataset/500km/modis_500km"
saveloc = "../data/classification_results/500km"

fc_img_paths = filter(r -> r != ".DS_Store", readdir(joinpath(dataloc, "falsecolor")))
land_mask_paths = replace.(fc_img_paths, "falsecolor"=>"landmask")

# read in image
for idx in 1:length(fc_img_paths)
    fc_img = Images.load(joinpath(dataloc, "falsecolor", fc_img_paths[idx]))
    land_mask = Gray.(Images.load(joinpath(dataloc, "landmask", land_mask_paths[idx]))) .> 0
    
    # classify
    classified_img = classifier(fc_img, land_mask);
    masks = Dict(k => classified_img .== classifier.key[k] for k in keys(classifier.key));
    
    # save
    save(joinpath(saveloc, 
            replace(
                replace(fc_img_paths[idx], "falsecolor"=>"classified"),
                    "tiff"=>"png")
            ),
        colorize_classification(classified_img)
    )
    
    # extract results for metrics, add to dataframe
    satellite = split(fc_img_paths[idx], ".")[2]
    case_idx = data.case .== split(fc_img_paths[idx], ".")[1] .&& (data.satellite .== satellite);
    df = data[case_idx, :]
    rc = round.(Int64, df[:, "row_center"])
    cc = round.(Int64, df[:, "col_center"])
    data[case_idx, :ift_classification] .= [classified_img[rc[ii], cc[ii]] for ii in 1:nrow(df)];
end

# At end of loop: get classification and make it plain text.
# Intentionally merge the bright and dark ice categories.
data[:, :ift_category] .= map(
    i -> Dict(0 => "land", 1 => "water", 2 => "ice", 3 => "ice", 4 => "cloud", -1 => Missing)[i], 
    data.ift_classification)
save("../data/classification_results/classification_results.csv", data)