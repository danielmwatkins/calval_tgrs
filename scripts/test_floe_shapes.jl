using Pkg;
Pkg.activate("../notebooks/calval")

using IceFloeTracker
using IceFloeTracker: load, regionprops_table, label_components, imshow, absdiffmeanratio, mismatch, addfloemasks!
using DataFrames, Statistics, CSV, Dates, Plots, Interpolations, Images

test_images_loc = "../data/test_images_tracker_setup/"

# convenience functions
greaterthan0(x) = x .> 0 # convert labeled image to boolean
greaterthan05(x) = x .> 0.5 # used for the image resize step

for fname in ["001-baffin_bay-20220911-aqua-labeled_floes.png",
              "022-barents_kara_seas-20060909-aqua-labeled_floes.png",
              "043-beaufort_sea-20190813-aqua-labeled_floes.png",
              "065-bering_chukchi_seas-20080507-aqua-labeled_floes.png",
              "086-east_siberian_sea-20060927-aqua-labeled_floes.png", # likely will throw error, this image is empty
              "107-greenland_sea-20210506-aqua-labeled_floes.png",
              "128-hudson_bay-20190415-aqua-labeled_floes.png",
              "148-laptev_sea-20110324-aqua-labeled_floes.png",
              "171-sea_of_okhostk-20090618-aqua-labeled_floes.png"]
    image = load(joinpath(test_images_loc, fname))
        
    # Add labels and get region properties
    labeled_image = label_components(image);
    props = regionprops_table(labeled_image);
    
    addfloemasks!(props, greaterthan0.(labeled_image));

    df = DataFrame(
                   floe_id=Int16[],
                   scale=Float64[],
                   rotation=Float64[],
                   area=Float64[],
                   convex_area=Float64[],
                   major_axis_length=Float64[],
                   minor_axis_length=Float64[],
                   adr_area=Float64[],
                   adr_convex_area=Float64[],
                   adr_major_axis_length=Float64[],
                   adr_minor_axis_length=Float64[],
                   est_rotation=Float64[],
                   mismatch=Float64[],
                   recall=Float64[],
                   normalized_sd=Float64[]
                   )
    floe_id = 1
    for floe_data in eachrow(props) # replace this list with iterating through rows of props and checking area
        if floe_data["area"] > 500
            init_floe = copy(floe_data["mask"])
            # pad the floe to avoid changing floe area relative to image size
            n = Int64(round(maximum(size(init_floe))))
            padded_init = collect(padarray(init_floe, Fill(0, (n, n), (n, n))))
            for rotation in range(-90, 90, 61)
                im_rotated = collect(imrotate(padded_init, deg2rad(rotation),
                                         axes(padded_init), method=BSpline(Constant())));
                for scale in [1, 0.5, 0.25]
                    if scale < 1
                        new_size = trunc.(Int, size(padded_init) .* scale)
                        im_scaled = greaterthan05(collect(imresize(padded_init, new_size)))
                        im_scaled_rotated = greaterthan05(collect(imresize(im_rotated, new_size)))
                    else
                        im_scaled = greaterthan05(copy(padded_init))
                        im_scaled_rotated = greaterthan05(copy(im_rotated))
                    end
                    
                    scaled_props = regionprops_table(label_components(im_scaled))
                    scaled_rotated_props = regionprops_table(label_components(im_scaled_rotated))
                    
                    # rotation estimate
                    mm, estimated_rotation = mismatch(im_scaled, im_scaled_rotated, mxrot=π/2)

                    un_rotated = collect(imrotate(im_scaled_rotated, -deg2rad(estimated_rotation),
                                         axes(im_scaled), method=BSpline(Constant())));
                    
                    # This should be close to right
                    # Ideally though the objects would be aligned at their centroids
                    
                    a_not_b = (im_scaled .> 0) .& (greaterthan05(un_rotated) .== 0);
                    b_not_a = (im_scaled .== 0) .& (greaterthan05(un_rotated) .> 0);
                    normalized_sd = sum(a_not_b .|| b_not_a) / scaled_rotated_props[1,:area]
                    
                    a_and_b = im_scaled .== greaterthan05(un_rotated)
                    recall = sum(a_and_b) / scaled_rotated_props[1,:area]
                    
                    push!(df, (floe_id,
                               scale,
                               rotation,
                               scaled_rotated_props[1,:area],
                               scaled_rotated_props[1,:convex_area],
                               scaled_rotated_props[1,:major_axis_length],
                               scaled_rotated_props[1,:minor_axis_length],
                               absdiffmeanratio(scaled_props[1,:area], scaled_rotated_props[1,:area]),
                               absdiffmeanratio(scaled_props[1,:convex_area], scaled_rotated_props[1,:convex_area]),
                               absdiffmeanratio(scaled_props[1,:major_axis_length], scaled_rotated_props[1,:major_axis_length]),
                               absdiffmeanratio(scaled_props[1,:minor_axis_length], scaled_rotated_props[1,:minor_axis_length]),
                               estimated_rotation,
                               mm,
                               recall,
                               normalized_sd
                               )) 
                   
                end # end scale loop
            end # end rotation loop
            floe_id += 1
        end # end "if size > threshold"
    end # end row loop
    
    print("Writing to file\n")
    CSV.write("../data/rotate_rescale/shape_rotate_rescale_"*fname[1:3]*".csv", df);
end