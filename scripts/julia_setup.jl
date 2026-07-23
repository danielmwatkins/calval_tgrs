# Sets up the cal-val environment for running the IFT scripts
# and initialize the Jupyter kernelß
using Pkg;
Pkg.activate("calval")
Pkg.add(; name="IceFloeTracker", rev="main")

Pkg.add(["IJulia",
        "DataFrames",
        "Plots",
        "CairoMakie",
        "CSVFiles",
        "Interpolations",
        "Images",
        "ImageSegmentation",
        "FileIO",
        "StatsBase" 
        ])
Pkg.update(; name="IceFloeTracker", rev="main")

using IceFloeTracker
using IJulia

Pkg.build()
Pkg.resolve()
Pkg.instantiate()
