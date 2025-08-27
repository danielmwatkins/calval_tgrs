# Sets up the cal-val environment for running the IFT scripts

using Pkg;
Pkg.activate("cal-val")
Pkg.add(["IJulia",
        "IceFloeTracker",
        "DataFrames",
        "CairoMakie",
        "CSV",
        "CSVFiles",
        "Interpolations",
        "Images",
        "ImageSegmentation",
        "FileIO"])
Pkg.build()
Pkg.resolve()
Pkg.instantiate()