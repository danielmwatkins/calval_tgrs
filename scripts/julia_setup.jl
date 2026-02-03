# Sets up the cal-val environment for running the IFT scripts

using Pkg;
Pkg.activate("calval")
Pkg.add(; name="IceFloeTracker", rev="main")

Pkg.add(["IJulia",
        "DataFrames",
        "CairoMakie",
        "CSVFiles",
        "Interpolations",
        "Images",
        "ImageSegmentation",
        "FileIO",
        "Graphs", # used for the segmentation merges
        "StatsBase" 
        ])
Pkg.update("IceFloeTracker")
using IceFloeTracker

Pkg.build()
Pkg.resolve()
Pkg.instantiate()
