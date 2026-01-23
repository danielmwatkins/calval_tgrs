# Sets up the cal-val environment for running the IFT scripts

using Pkg;
Pkg.activate("cal-val")
Pkg.add(; name="IceFloeTracker", rev="main")

Pkg.add(["IJulia",
        "DataFrames",
        "CairoMakie",
        "CSV",
        "CSVFiles",
        "Interpolations",
        "Images",
        "ImageSegmentation",
        "FileIO"])
Pkg.update("IceFloeTracker")
using IceFloeTracker

Pkg.build()
Pkg.resolve()
Pkg.instantiate()