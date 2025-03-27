using Pkg;
Pkg.activate("calval")

using IceFloeTracker
using IceFloeTracker: load, regionprops_table, label_components, imshow, absdiffmeanratio, mismatch, addfloemasks!
using DataFrames, Statistics, CSV, Dates, Plots, Interpolations, Images

test_images_loc = "../../eval_seg/data/validation_images/binary_floes/"

# convenience functions
greaterthan0(x) = x .> 0 # convert labeled image to boolean
greaterthan05(x) = x .> 0.5 # used for the image resize step
imrotate_bin(x, r) = greaterthan05(collect(imrotate(x, deg2rad(r), method=BSpline(Constant()))))

files = readdir(test_images_loc)
files = [f for f in files if occursin("aqua", f)]
    
for fname in files
    image = load(joinpath(test_images_loc, fname))
        
    # Add labels and get region properties
    labeled_image = label_components(image);
    props = regionprops_table(labeled_image);
    
    addfloemasks!(props, greaterthan0.(labeled_image));

    df = DataFrame(
                   floe_id=Int16[],
                   rotation=Float64[],
                   area=Float64[],
                   convex_area=Float64[],
                   major_axis_length=Float64[],
                   minor_axis_length=Float64[],
                   adr_area=Float64[],
                   adr_convex_area=Float64[],
                   adr_major_axis_length=Float64[],
                   adr_minor_axis_length=Float64[]
                   )
    floe_id = 1
    for floe_data in eachrow(props)
        if floe_data["area"] > 50
            im_init = copy(floe_data["mask"])
            
            # pad the floe to avoid changing floe area relative to image size
            n = Int64(round(maximum(size(im_init))))
            im_padded = collect(padarray(im_init, Fill(0, (n, n), (n, n))))
            
            # recalculate properties -- should be same as init except centroid/bbox
            init_props = regionprops_table(label_components(im_padded))
            for rotation in range(-45, 45, 31)
                im_rotated = imrotate_bin(im_padded, rotation)
                rotated_props = regionprops_table(label_components(im_rotated))

                push!(df, (floe_id,
                           rotation,
                           rotated_props[1,:area],
                           rotated_props[1,:convex_area],
                           rotated_props[1,:major_axis_length],
                           rotated_props[1,:minor_axis_length],
                           absdiffmeanratio(floe_data["area"], rotated_props[1,:area]),
                           absdiffmeanratio(floe_data["convex_area"], rotated_props[1,:convex_area]),
                           absdiffmeanratio(floe_data["major_axis_length"], rotated_props[1,:major_axis_length]),
                           absdiffmeanratio(floe_data["minor_axis_length"], rotated_props[1,:minor_axis_length]),
                           )) 
            end
        end
        floe_id += 1
    end
    
    
    print("Writing to file\n")
    CSV.write("../data/rotation_test/"*fname[1:3]*"-shape-rotation.csv", df);
end