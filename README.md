# calval_tgrs
Calibration and validation code for the IFT TGRS manuscript
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
- 