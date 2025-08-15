# calval_tgrs
Calibration and validation code for the IFT TGRS manuscript. Setup:
- Download the ice_floe_validation_dataset and put it in the same directory as calval_tgrs, so relative imports will work
- Install the calval environment
- Install the latest version of Julia
- Run the julia_setup.jl script

# data
The main source of data is the random sample dataset, which is in the folder `eval_seg`.

# components
## cloud mask
- questions about aligning the dataset so that spatial error can be calculated
- scatter plot based on the band 7 to band 2 scatter plot

# tbd
- finish analysis of IABP data for distance measure
- use the rotation test data, filtering by test/train, and getting the maximum (ADR, SD) and minimum (rho) over each.
- add psi-s correlation and perimeter to the rotation test script
- add a script to carry out the matched pairs analysis


# Calibrating the tracking module
The calibration and initial validation of the tracking module is carried out using the floes in the validation dataset. We need to understand how the tracking performance is affected by uncertainty in the floe boundary.  For this reason we need to take the set of tracked floes, and select floes based on the confidence in the boundary. 
- Fully clear sky floes: Floe and its boundary (dilation by 2 pixels) contains no (or less than x%) pixels flagged as cloudy by the IFT cloud mask
- Partially obscured floes: Floe and its boundary have cloud fraction between (min, max).
- Fully obscured floes: Floe and its boundary have cloud fraction greater than (max).

Coding steps
1. Add script to generate floe property tables and save the tables in the calval_tgrs folder. Added, but still need a few details -- need the pass times, and need the
2. Add script to add pass-times to the property tables.
3. DONE: Add cloud fraction property to the exported property tables
4. Merge exported property tables with the IFT exported "matched pairs" tests
5. In the visualizations for ADR, SD, and psi-s correlation, distinguish the cloud-obscured and clear-sky floes
6. Update the regionprops_table to include calculations of floe-average cloud fraction, band 7 and band 2 reflectance, etc.
7. Set up notebook with the internal steps of the tracking algorithm.
   * Check: are we exiting the comparison when certain tests fail?
9. Generate figure with example of tracking progress:
   * Distance circle outlining floes with centroids within possible distance
   * Outlining floes that have all ADRs below the relevant thresholds
   * Outlining floes that passing correlation
   * Normalized shape difference curves for remaining passing floes
10. Describe the process for minimizing the error (distance, ADRs, 1 - correlation, shape difference) across the set of floes, allowing for floes to have no match when appropriate.
11. Figure showing candidate matches and final matches.