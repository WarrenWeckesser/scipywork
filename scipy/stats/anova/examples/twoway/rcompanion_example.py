import numpy as np
from anova import anova_twoway_from_x_y_values

# See https://rcompanion.org/rcompanion/d_08.html
acttext = """
  1 male    ff        1.884
  2 male    ff        2.283
  3 male    fs        2.396
  4 female  ff        2.838  
  5 male    fs        2.956
  6 female  ff        4.216
  7 female  ss        3.620
  8 female  ff        2.889  
  9 female  fs        3.550
 10 male    fs        3.105
 11 female  fs        4.556
 12 female  fs        3.087
 13 male    ff        4.939
 14 male    ff        3.486
 15 female  ss        3.079
 16 male    fs        2.649
 17 female  fs        1.943
 19 female  ff        4.198
 20 female  ff        2.473
 22 female  ff        2.033 
 24 female  fs        2.200
 25 female  fs        2.157
 26 male    ss        2.801
 28 male    ss        3.421
 29 female  ff        1.811
 30 female  fs        4.281
 32 female  fs        4.772
 34 female  ss        3.586
 36 female  ff        3.944
 38 female  ss        2.669
 39 female  ss        3.050
 41 male    ss        4.275 
 43 female  ss        2.963
 46 female  ss        3.236
 48 female  ss        3.673
 49 male    ss        3.110
 """

actfields = [w.split()[1:] for w in acttext.strip().splitlines()]
sex, genotype, activity = list(zip(*actfields))
activity = [float(s) for s in activity]

result = anova_twoway_from_x_y_values(x=sex, y=genotype, values=activity)
print(result)