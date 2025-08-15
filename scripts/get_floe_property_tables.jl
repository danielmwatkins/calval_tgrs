# Extract features from validated images, add information on cloud fraction

saveloc = "../data/floe_property_tables/"
dataloc = "../../ice_floe_validation_dataset/"

"""Access the matched pairs from the validation dataset and compute similarity metrics"""

using Pkg
Pkg.activate("../scripts/cal-val")

using IceFloeTracker
using DataFrames, CSV, Images

saveloc = "../data/floe_property_tables/"
ice_floe_database_loc =  "/Users/dwatkin2/Documents/research/manuscripts/cal-val_ice_floe_tracker/ice_floe_validation_dataset/"
labeled_image_loc = "data/validation_dataset/labeled_floes/"
truecolor_image_loc = "data/modis/truecolor/"
falsecolor_image_loc = "data/modis/falsecolor/"

# Parameters
cloud_mask_settings = (
    prelim_threshold=53.0/255.,
    band_7_threshold=130.0/255.,
    band_2_threshold=169.0/255.,
    ratio_lower=0.0,
    ratio_offset=0.0,
    ratio_upper=0.53
)
min_area = 50
properties = ["label", "area", "bbox", "centroid", "convex_area", "major_axis_length", "minor_axis_length",
              "orientation", "perimeter", "perimeter_crofton"]
column_order = [:label, :row_centroid, :col_centroid,:min_row, :min_col, :max_row, :max_col,
                :area, :convex_area, :major_axis_length, :minor_axis_length, :orientation,
                :perimeter, :perimeter_crofton, :cloud_fraction, :band_7_reflectance, :band_2_reflectance]

LACM = IceFloeTracker.LopezAcostaCloudMask(cloud_mask_settings...)

files = readdir(joinpath(ice_floe_database_loc, labeled_image_loc))
files = [f for f in files if occursin("tiff", f)]; 

for file in files
    labeled_image = channelview(Int64.(load(joinpath(ice_floe_database_loc,  labeled_image_loc, file))))
    case, region, date, satellite, suffix = split(file, "-")    
    falsecolor_image = load(joinpath(ice_floe_database_loc,
                                     falsecolor_image_loc,
                                     join([case, region, "100km", date], "-")*"."*satellite*".falsecolor.250m.tiff"))
    
    df = IceFloeTracker.regionprops_table(labeled_image; properties=properties)
    df = filter(:area => a -> a .> min_area, df)
    
    if nrow(df) > 0
        
        # Get segment means
        cloudmask = IceFloeTracker.create_cloudmask(falsecolor_image, LACM)
        fc_image_data = SegmentedImage(falsecolor_image, labeled_image)
        cm_image_data = SegmentedImage(cloudmask, labeled_image)

        # Add segment means to the dataframe
        df[!,:cloud_fraction] .= [cm_image_data.segment_means[x] for x in df[!, :label]]
        df[!,:band_7_reflectance] .= [red(fc_image_data.segment_means[x]) for x in df[!, :label]]
        df[!,:band_2_reflectance] .= [green(fc_image_data.segment_means[x]) for x in df[!, :label]];
        df[!,:band_1_reflectance] .= [blue(fc_image_data.segment_means[x]) for x in df[!, :label]];
        df = round.(df, digits=2)
        df[!,:area] = convert.(Int64,df[!,:area])
        
        CSV.write(joinpath(saveloc, satellite, replace(file, "labeled_floes.tiff" => "floe_properties.csv")),
          df)
    end
end