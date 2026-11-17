using Pkg
Pkg.activate("../scripts/calval")

using CairoMakie
using CSV
using DataFrames
using Dates
using IceFloeTracker
using Images
using StatsBase

data = DataFrame(CSV.File("../data/calval_data_500km_samples.csv"))
# Scale inputs to between 0 and 1
for col in [:b7_mean, :b2_mean, :b1_mean]
    data[:, col] .= data[:, col] ./ 255
end

data[:, :parsed_date] = Date.(string.(data[:, :date]), "yyyymmdd")
data[:, :month] = month.(data.parsed_date);

region_numbers = Dict(
 "baffin_bay" => 4,
 "barents_kara_seas" => 6,
 "beaufort_sea" => 2,
 "bering_chukchi_seas" => 1,
 "east_siberian_sea" => 9,
 "greenland_sea" => 5,
 "hudson_bay" => 3,
 "laptev_sea" => 7,
 "sea_of_okhostk" => 8
)

group_means = combine(groupby(data, ["month", "init_cat"]), 
    nrow,
    :b1_mean => mean => :b1_mean,
    :b2_mean => mean => :b2_mean,
    :b7_mean => mean => :b7_mean,
    :b1_mean => median => :b1_median,
    :b2_mean => median => :b2_median,
    :b7_mean => median => :b7_median,
    :b1_mean => (r -> percentile(r, 1)) => :b1_p01,
    :b2_mean => (r -> percentile(r, 1)) => :b2_p01,
    :b7_mean => (r -> percentile(r, 1)) => :b7_p01,
    :b1_mean => (r -> percentile(r, 99)) => :b1_p99,
    :b2_mean => (r -> percentile(r, 99)) => :b2_p99,
    :b7_mean => (r -> percentile(r, 99)) => :b7_p99, 
    :b1_mean => (r -> percentile(r, 25)) => :b1_p25,
    :b2_mean => (r -> percentile(r, 25)) => :b2_p25,
    :b7_mean => (r -> percentile(r, 25)) => :b7_p25,
    :b1_mean => (r -> percentile(r, 75)) => :b1_p75,
    :b2_mean => (r -> percentile(r, 75)) => :b2_p75,
    :b7_mean => (r -> percentile(r, 75)) => :b7_p75
    );

f = CairoMakie.Figure(size=(1000,600))

counts = combine(groupby(data, ["region", "init_cat"]), nrow);
counts_dict = Dict()
push!(counts_dict, "region"=>counts[counts.init_cat .== "none","region"])
for category in ["clear_water", "clear_ice", "cloudy_water", "cloudy_ice", "none"]
    push!(counts_dict, category => counts[counts.init_cat .== category, "nrow"])
end

counts_df = DataFrame(counts_dict)
counts_df.region_number = [region_numbers[r] for r in counts_df.region]

factor_levels = Dict(
    "clear_ice" => 1,
    "clear_water" => 2,
    "cloudy_ice" => 3,
    "cloudy_water" => 4,
    "none" => 5)
melted_df = stack(counts_df, [:clear_ice, :clear_water, :cloudy_ice, :cloudy_water, :none], [:region_number, :region])
melted_df[:, :group] = map(r -> factor_levels[r], melted_df.variable);

colors = CairoMakie.categorical_colors(:Dark2_8, 8)
colors = colors[[6, 7, 3, 1, 8, 5]]

ax = CairoMakie.Axis(f[1,1], 
    xticks=1:9,
    yticks=([0, 0.5e4, 1e4, 1.5e4, 2e4, 2.5e4], ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5"]),
    xlabel="Region Number",
    titlesize=16,
    title="Sample Category", ylabel="Counts (×10⁴)")
barplot!(ax,
    melted_df.region_number,
    melted_df.value, 
    stack=Vector(melted_df.group),
    color=[colors[x] for x in Vector(melted_df.group)],
)

counts = combine(groupby(data, ["month", "init_cat"]), nrow);
counts_dict = Dict()
push!(counts_dict, "month"=>counts[counts.init_cat .== "none","month"])
for category in ["clear_water", "clear_ice", "cloudy_water", "cloudy_ice", "none"]
    push!(counts_dict, category => counts[counts.init_cat .== category, "nrow"])
end

counts_df = DataFrame(counts_dict)
melted_df = stack(counts_df, [:clear_ice, :clear_water, :cloudy_ice, :cloudy_water, :none], [:month])
melted_df[:, :group] = map(r -> factor_levels[r], melted_df.variable);

ax = CairoMakie.Axis(f[2, 1], 
    xticks=(3:9, ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]),
    titlesize=16,
    yticks=([0, 0.5e4, 1e4, 1.5e4, 2e4, 2.5e4], ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5"]),
    title="Sample Category", ylabel="Counts (×10⁴)")
barplot!(ax,
    melted_df.month,
    melted_df.value, 
    stack=Vector(melted_df.group),
    color=[colors[x] for x in Vector(melted_df.group)],
)

labels = keys(factor_levels)
elements = [[MarkerElement(color = colors[factor_levels[i]],
            marker = :rect, markersize = 15)]
            for i in labels
            ]
Legend(f[1, 4], elements, [titlecase(replace(l, "_"=>" ")) for l in labels], nbanks=1)

colors = CairoMakie.categorical_colors(:seaborn_colorblind, 3)

# Clear-sky ice
ax = CairoMakie.Axis(f[1,2], xlabel="",  titlesize=16,
    ylabel="Reflectance", xticks=(3:9, ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]), yticks=0:0.2:1, limits=(2.75, 9.25, 0, 1), title="Sea Ice")
category = "clear_ice"
df = subset(group_means, :init_cat => r -> r .== category)
sort!(df, :month)
for (idx, band) in enumerate(["b1", "b2", "b7"])
    x = df[!, :month] # .+ (1 - idx)/20
    rangebars!(ax, x, df[!, band*"_p25"], df[!, band*"_p75"], linewidth=5,
        color=colors[idx], whiskerwidth=0)
    rangebars!(ax, x, df[!, band*"_p01"], df[!, band*"_p99"], linewidth=1,
        color=colors[idx], whiskerwidth=10)
    scatter!(ax, x, df[!, band*"_mean"],
        strokecolor=colors[idx], color=:white, strokewidth=1, marker=:diamond)
    scatterlines!(ax, x, df[!, band*"_median"], linestyle=:solid,
        strokecolor=colors[idx], markercolor=:white, strokewidth=1)

end

# Clear-sky water
ax = CairoMakie.Axis(f[1,3], xlabel="",
    ylabel="Reflectance",  titlesize=16,
    xticks=(3:9, ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]), yticks=0:0.1:0.5, limits=(2.75, 9.25, 0, 0.5), title="Water")
category = "clear_water"
df = subset(group_means, :init_cat => r -> r .== category)
sort!(df, :month)
for (idx, band) in enumerate(["b1", "b2", "b7"])
    x = df[!, :month] .+ (1 - idx)/20
    scatter!(ax, x, df[!, band*"_mean"], 
        strokecolor=colors[idx], color=:white, strokewidth=1, marker=:diamond)
    scatterlines!(ax, x, df[!, band*"_median"], linestyle=:solid, 
        strokecolor=colors[idx], markercolor=:white, strokewidth=1)
    rangebars!(ax, x, df[!, band*"_p25"], df[!, band*"_p75"], linewidth=5,
        color=colors[idx], whiskerwidth=0)
    rangebars!(ax, x, df[!, band*"_p01"], df[!, band*"_p99"], linewidth=1,
        color=colors[idx], whiskerwidth=10)
end

# For clouds, we show the clouds over ice and water separately
ax = CairoMakie.Axis(f[2,2], xlabel="", titlesize=16,
    ylabel="Reflectance", xticks=(3:9, ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]),
    yticks=0:0.2:1, limits=(2.75, 9.25, 0, 1.),
    title="Cloud-Covered Ice")

category = "cloudy_ice"
df = subset(group_means, :init_cat => r -> r .== category)
sort!(df, :month)
for (idx, band) in enumerate(["b1", "b2", "b7"])
    x = df[!, :month] .+ (1 - idx)/20
    rangebars!(ax, x, df[!, band*"_p25"], df[!, band*"_p75"], linewidth=5,
        color=colors[idx], whiskerwidth=0)
    rangebars!(ax, x, df[!, band*"_p01"], df[!, band*"_p99"], linewidth=1,
        color=colors[idx], whiskerwidth=10)
    scatter!(ax, x, df[!, band*"_mean"], 
        strokecolor=colors[idx], color=:white, strokewidth=1, marker=:diamond)
    scatterlines!(ax, x, df[!, band*"_median"], linestyle=:solid,
        strokecolor=colors[idx], markercolor=:white, strokewidth=1)
    
end
ax = CairoMakie.Axis(f[2,3], xlabel="", titlesize=16,
    xticks=(3:9, ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]),
    ylabel="Reflectance", yticks=0:0.2:1, limits=(2.75, 9.25, 0, 1.),
    title="Cloud-Covered Water")
category = "cloudy_water"
df = subset(group_means, :init_cat => r -> r .== category)
sort!(df, :month)
for (idx, band) in enumerate(["b1", "b2", "b7"])
    x = df[!, :month] .+ (1 - idx)/20
    rangebars!(ax, x, df[!, band*"_p25"], df[!, band*"_p75"], linewidth=5,
        color=colors[idx], whiskerwidth=0)
    rangebars!(ax, x, df[!, band*"_p01"], df[!, band*"_p99"], linewidth=1,
        color=colors[idx], whiskerwidth=10)
    scatter!(ax, x, df[!, band*"_mean"], 
        strokecolor=colors[idx], color=:white, strokewidth=1, marker=:diamond)
    scatterlines!(ax, x, df[!, band*"_median"], linestyle=:solid,
        strokecolor=colors[idx], markercolor=:white, strokewidth=1)

end

elem_p1p99 = [LineElement(color = :black, linestyle = :solid)]
elem_iqr = [LineElement(color = :black, linestyle = :solid, linewidth=5)]
elem_median = MarkerElement(color = :white, strokecolor=:black, strokewidth=1, marker = :circle, markersize = 15)
elem_mean = MarkerElement(color = :white, strokecolor=:black, strokewidth=1, marker = :diamond, markersize = 15)
elem_band_1 = [LineElement(color = colors[1], linestyle = :solid),
          MarkerElement(color = colors[1], marker = :circle, markersize = 15,
          strokecolor = :black)]
elem_band_2 = [LineElement(color = colors[2], linestyle = :solid),
          MarkerElement(color = colors[2], marker = :circle, markersize = 15,
          strokecolor = :black)]
elem_band_3 = [LineElement(color = colors[3], linestyle = :solid),
          MarkerElement(color = colors[3], marker = :circle, markersize = 15,
          strokecolor = :black)]
CairoMakie.Legend(f[2, 4],
    [
     elem_band_1, elem_band_2, elem_band_3, elem_median, elem_mean, elem_p1p99, elem_iqr],
    ["Band 1", "Band 2", "Band 7", "Median", "Mean", "1-99 Range", "25-75 Range",  ], nbanks=1)


labels = ["a", "b", "c", "d", "e", "f"]
for (label, pos) in zip(labels, [f[1, 1], f[1, 2], f[1, 3], f[2, 1], f[2, 2], f[2, 3]])
    Label(pos[1, 1, TopLeft()], label,
        fontsize = 16,
        font = :bold,
        padding = (0, 1, 8, 0), # Adds space to the right and bottom
        halign = :left)
end

save("../figures/fig_03_500km_sample_summary.png", f)
f