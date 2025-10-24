using Pkg
Pkg.activate("cal-val")
using IceFloeTracker
using IceFloeTracker: AbstractCloudMaskAlgorithm, fill_holes
using Images

save_loc = "../data/ift_cloud_mask/"
data_loader = Watkins2025GitHub(; ref="25ba4d46814a5423b65ad675aaec05633d17a37e")

cloud_mask_settings = (
    prelim_threshold=53.0/255.,
    band_7_threshold=130.0/255.,
    band_2_threshold=170.0/255.,
    ratio_lower=0.0,
    ratio_offset=0.0,
    ratio_upper=0.52
)

@kwdef struct Watkins2025CloudMask <: AbstractCloudMaskAlgorithm
    prelim_threshold::Float64 = 0.21
    band_7_threshold::Float64 = 0.51
    band_2_threshold::Float64 = 0.66
    ratio_lower::Float64 = 0.0
    ratio_offset::Float64 = 0.0
    ratio_upper::Float64 = 0.53
    marker_strel = strel_box((7,7))
    opening_strel = strel_diamond((3,3))
end

# When update is merged, we can call this directly.
function (f::Watkins2025CloudMask)(img::AbstractArray{<:Union{AbstractRGB,TransparentRGB}})
    init_cloud_mask = LopezAcostaCloudMask(f.prelim_threshold,
                                           f.band_7_threshold,
                                           f.band_2_threshold,
                                           f.ratio_lower,
                                           f.ratio_offset,
                                           f.ratio_upper)(img)
    markers = opening(init_cloud_mask, f.marker_strel)
    reconstructed = mreconstruct(dilate, markers, init_cloud_mask, f.opening_strel)
    smoothed = opening(reconstructed, f.opening_strel)
    filled = fill_holes(smoothed)
    return filled
end

old_cmask = LopezAcostaCloudMask();
new_init = LopezAcostaCloudMask(cloud_mask_settings...);
new_cmask = Watkins2025CloudMask();

dataset = data_loader(c -> c.case_number < 190);
for case in dataset
    name = case.name
    fc = RGB.(case.modis_falsecolor)
    lm = Gray.(case.modis_landmask) .> 0
    old = old_cmask(fc)
    new_raw = new_init(fc)
    new_clean = new_cmask(fc)
    save(joinpath(save_loc, "lopez_acosta", name*"-cloudmask.png"), old)
    save(joinpath(save_loc, "initial", name*"-cloudmask.png"), new_raw)
    save(joinpath(save_loc, "cleaned", name*"-cloudmask.png"), new_clean)
end