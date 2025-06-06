import proplot as pplt
import numpy as np
import pandas as pd

def draw_mask(t0=110,  t7=200, t2=190, r_lower=0, r_upper=0.75, variant="LSW2019"):
    """Returns list of x/y pairs to plot mask partition curve. r_lower not used at all for now."""
    if variant=="LSW2019":
        intersect_t2 = (t2, r_upper*t2)
        intersect_t7 = (t7/r_upper, t7)
        intersect_t0 = (t0/r_upper, t0)
        # If t7 is below t0, then no other unmasking can happen
        if t7 <= t0:
            # print('Case 1')
            x = [0, 255]
            y = [t0, t0]
            return x, y

        # If t2*r_upper is greater than t7, then the ratio is not used
        elif t2*r_upper >= t7:
            # print('Case 2')
            x = [0, t2, t2, 255]
            y = [t0, t0, t7, t7]
            return x, y

        # If the intersection of t7 and the ratio line is outside the range
        # of pixel intensities, then the t7 threshold is not used.
        elif t7/r_upper >= 255: 
            
            # If the intersection between the ratio and the t2 threshold
            # is below the t0 threshold, we get a three-point line
            if r_upper*t2 <= t0: 
                # print('Case 3')
                x = [0, t0/r_upper, 255]
                y = [t0, t0, 255*r_upper]
                return x, y

            # Otherwise, we get a four-point curve with vertical line at the t2 threshold
            # This is the one used by default. 
            else:
                # print('Case 4')
                x = [0, t2, t2, 255]
                y = [t0, t0, r_upper*t2, 255*r_upper]
                return x, y
                
        # Finally we look at where the t7 threshold matters
        else:

            # If the t2 threshold and the ratio line intersect below the t0
            # threshold, then t2 threshold doesn't do anything and we get
            # a four-point curve with horizontal line set by t7
            if r_upper*t2 <= t0:
                # print('Case 5')
                x = [0, t0/r_upper, t7/r_upper, 255]
                y = [t0, t0, t7, t7]
                return x, y

            # Otherwise, we get the only case where all the thresholds matter.
            else:
                # print('Case 6')
                # Five point curve with vertical and horizontal lines
                x = [0, t2, t2, t7/r_upper, 255]
                y = [t0, t0, r_upper*t2, t7, t7]
                return x, y
        
                # Other possibility: what if t0 = 0?

fig, ax = pplt.subplots()

# Case 1
x, y = draw_mask(t7=50, t2=50, t0=75, r_upper=0.4)
print(x, y)
ax.plot(x, y, label='Unmasked region all below the t0 curve')


# Case 2: t2, t7 intersection is below ratio line, ratio is not used.
x, y = draw_mask(t7=100, t2=160, t0=90)
print(x, y)
ax.plot(x, y, label='(t2, t7) below ratio line')

# Case 3 
x, y = draw_mask(t7=255, t2=160, t0=150, r_upper=0.9)
# print(x, y)
ax.plot(x, y, label='Only t0 and ratio matter')

# Case 4: Default settings. Only ratio, t0, and t2 matter.
x, y = draw_mask()
ax.plot(x, y, label='Default LSW2019')

# Case 5: t2 not used, other settings in effect
x, y = draw_mask(t0=175, t2=0, r_upper=1, t7=225)
ax.plot(x, y, label='t2 below usable limit')

# Case 6: All settings used
x, y = draw_mask(t0=25, t2=150, t7=60, r_upper=0.3)
ax.plot(x, y, label='All settings used')


ax.format(ylim=(0, 255), xlim=(0, 255))
ax.legend(ncols=1, loc='r')

fig.save('../figures/example_cloud_mask_curves.png', dpi=300)