"""Run the preprocessing script on the validation dataset to produce morphological residue"""

using Pkg
# Specifying an environment makes it easier to work with package versions
home_dir = "../"
Pkg.activate(joinpath(home_dir, "notebooks/calval"))

using IceFloeTracker
using Images
using Dates
using DataFrames
using CSV

# get case data
df = DataFrame(CSV.File("../data/validation_dataset_testtrain_split.csv"));

# set parameters used in preprocessing functions
cloud_mask_settings = (
    prelim_threshold=53.0/255.,
    band_7_threshold=130.0/255.,
    band_2_threshold=169.0/255.,
    ratio_lower=0.0,
    ratio_upper=0.53
)

adjust_gamma_params = (gamma=1.5, gamma_factor=1.3, gamma_threshold=220)

structuring_elements = (
    se_disk1=collect(IceFloeTracker.MorphSE.StructuringElements.strel_diamond((3, 3))),
    se_disk2=IceFloeTracker.se_disk2(),
    se_disk4=IceFloeTracker.se_disk4(),
)

unsharp_mask_params = (radius=10.0, amount=1.5, factor=255.0)

brighten_factor = 0.1;

adapthisteq_params = (
    white_threshold=25.5, entropy_threshold=4, white_fraction_threshold=0.4
)

# set locations
data_dir = "../../ice_floe_validation_dataset/"
save_dir = "../data/validation_dataset/"
for row in eachrows(df)
    case_number = lpad(row[:case_number], 3, "0")
    region = row[:region]
    date = Dates.format(row[:start_date], "yyyymmdd")
    
    lm_filename = join([case_number, region, "100km", date], "-")*"."*join([row[:satellite], "landmask", "250m", "tiff"], ".")
    tc_filename = join([case_number, region, "100km", date], "-")*"."*join([row[:satellite], "truecolor", "250m", "tiff"], ".")
    fc_filename = join([case_number, region, "100km", date], "-")*"."*join([row[:satellite], "falsecolor", "250m", "tiff"], ".")
    cm_savename = join([case_number, region, "100km", date], "-")*"."*join([row[:satellite], "cloudmask", "250m", "tiff"], ".")
    mr_savename = join([case_number, region, "100km", date], "-")*"."*join([row[:satellite], "morphed_residue", "250m", "tiff"], ".")
    
    
    
    lm_image = float64.(RGB.(load(joinpath(data_dir, "modis", "landmask", lm_filename))))
    fc_image = float64.(RGB.(load(joinpath(data_dir, "modis", "falsecolor", fc_filename))))

    # generate cloudmask
    cloudmask = IceFloeTracker.create_cloudmask(fc_image; cloud_mask_settings...)
    fc_img_cloudmasked = IceFloeTracker.apply_cloudmask(fc_image, cloudmask)

    # treat image as a single tile
    prelim_sizes = size(tc_image) .÷ 1
    tiles = IceFloeTracker.get_tiles(tc_image, prelim_sizes[1] + 1);

    # generate band 7 image for equalized hist adjustment
    clouds_red = IceFloeTracker.to_uint8(float64.(red.(fc_img_cloudmasked) .* 255))
    clouds_red[.!landmask.dilated] .= 0;
    rgbchannels = IceFloeTracker._process_image_tiles(
        tc_image, clouds_red, tiles, adapthisteq_params...);
    
    equalized_gray = IceFloeTracker.rgb2gray(rgbchannels);

    # apply cloudmask
    masks = [f.(fc_img_cloudmasked) .== 0 for f in [red, green, blue]]
    combo_mask = reduce((a, b) -> a .& b, masks)
    equalized_gray[.!cloudmask] .= 0;

    # sharpen and reconstruct
    sharpened = IceFloeTracker.to_uint8(IceFloeTracker.unsharp_mask(equalized_gray, unsharp_mask_params...))
    equalized_gray_sharpened_reconstructed = IceFloeTracker.reconstruct(
            sharpened, structuring_elements.se_disk1, "dilation", true
        )
    equalized_gray_sharpened_reconstructed[.!landmask.dilated] .= 0;

    # reconstruct without sharpening
    equalized_gray_reconstructed = deepcopy(equalized_gray)
    equalized_gray_reconstructed[.!landmask.dilated] .= 0
    equalized_gray_reconstructed = IceFloeTracker.reconstruct(
        equalized_gray_reconstructed, structuring_elements.se_disk4, "dilation", true
    )
    equalized_gray_reconstructed[.!landmask.dilated] .= 0;

    # brighten green channel
    gammagreen = @view rgbchannels[:, :, 2];
    brighten = IceFloeTracker.get_brighten_mask(equalized_gray_reconstructed, gammagreen)
    equalized_gray[.!landmask.dilated] .= 0
    equalized_gray .= IceFloeTracker.imbrighten(equalized_gray, brighten, brighten_factor)

    # compute morphological residue
    morphed_residue = clamp.(equalized_gray - equalized_gray_reconstructed, 0, 255)

    # apply gamma correction
    equalized_gray_sharpened_reconstructed_adjusted = IceFloeTracker.imcomplement(
        IceFloeTracker.adjustgamma(equalized_gray_sharpened_reconstructed, adjust_gamma_params.gamma))
    adjusting_mask = equalized_gray_sharpened_reconstructed_adjusted .> adjust_gamma_params.gamma_threshold
    morphed_residue[adjusting_mask] .= IceFloeTracker.to_uint8.(
        morphed_residue[adjusting_mask] .* adjust_gamma_params.gamma_factor);

    # save results to file
    Images.save(joinpath(save_loc, "morphological_residue", mr_savename)),
        Gray.(morphed_residue./255))
    Images.save(joinpath(save_loc, "cloudmask", cm_savename)),
        Gray.(cloudmask))
end

