# Sets up the cal-val environment for running the IFT scripts

using Pkg;
Pkg.activate("cal-val")
Pkg.add(["IJulia", "IceFloeTracker", "DataFrames", "CairoMakie", "CSV", "Interpolations", "Images", "ImageSegmentation"])
Pkg.build()
Pkg.resolve()
Pkg.instantiate()