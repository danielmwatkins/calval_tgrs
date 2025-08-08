using Pkg
Pkg.activate("cal-val")
using IceFloeTracker
using CSV

function make_rectangle(area, aspect_ratio, angle_degrees)
    width = round(sqrt(area / aspect_ratio))
    height = round(aspect_ratio * width)
    box = Int32(maximum([width, height]) * 2)
    R = falses((box, box))
    center = Int32(round(box/2))
    hdims = Int64(round(center - height/2)):Int64(round(center + height/2 - 1))
    wdims = Int64(round(center - width/2)):Int64(round(center + width/2 - 1))
    R[hdims, wdims] .= true
    return IceFloeTracker.imrotate_bin_clockwise_degrees(R, angle_degrees)
end

function mismatch_temp(fixed::AbstractArray, moving::AbstractArray, test_angles::AbstractArray)
    shape_differences = IceFloeTracker.shape_difference_rotation(
        fixed, moving, test_angles; imrotate_function=IceFloeTracker.imrotate_bin_clockwise_degrees
    )
    best_match = argmin((x) -> x.shape_difference, shape_differences)
    rotation_degrees = best_match.angle
    normalized_area = (sum(fixed) + sum(moving)) / 2
    normalized_mismatch = best_match.shape_difference / normalized_area
    return (mm=normalized_mismatch, rot=rotation_degrees, sd=best_match.shape_difference)
end


area = [50, 100, 500, 1000, 2000, 5000, 10000, 20000]
aspect = [1.1, 1.5, 2., 2.5]
n = 200
n_cases = n * length(area) * length(aspect)
areas = zeros(n_cases)
aspects = zeros(n_cases)
floe1_theta = zeros(n_cases)
floe2_theta = zeros(n_cases)
recovered_theta = zeros(n_cases)
minimum_sd = zeros(n_cases)
row_idx = 0
@time begin
    for A in area
        for AR in aspect
            for idx in range(1, n)
                global row_idx += 1
                theta1, theta2 = 45 .* (rand(2) .- 0.5)
                floe1_theta[row_idx] = theta1
                floe2_theta[row_idx] = theta2
                R1 = make_rectangle(A, AR, theta1)
                R2 = make_rectangle(A, AR, theta2)
                mm, rot, sd = mismatch_temp(R1, R2, -100:1:100) # maximum difference between angles is 90
                recovered_theta[row_idx] = rot
                minimum_sd[row_idx] = sd
                areas[row_idx] = A
                aspects[row_idx] = AR
            end
        end
    end
end

results = DataFrame(
    area=areas,
    aspect=aspects,
    floe1_theta=floe1_theta,
    floe2_theta=floe2_theta,
    recovered_theta=recovered_theta,
    minimum_sd=minimum_sd)
results[:, "true_theta"] = floe2_theta .- floe1_theta;
results[:, "perimeter"] .= 0

for A in area
    for AR in aspect
        # using estimated perimeter rather than exact
        perimeter = IceFloeTracker.regionprops_table(
            label_components(make_rectangle(A, AR, 0)),
            properties=["perimeter"])[1,"perimeter"]
        results[results.area .== A .&& results.aspect .== AR, :perimeter] .= perimeter
    end
end

CSV.write("../data/rotation_test/rectangle-rotation-shape_difference.csv", results);