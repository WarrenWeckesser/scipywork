# ANOVA example from Section 23.6
# of Draper & Smith, Applied Regression Analysis, 3rd ed.
# The data is from Table 23.4, page 488.

import numpy as np
from anova import anova_twoway_balanced


# `rates` is an array with shape (4, 3, 2).  It holds the production rate
# data from Table 23.4 of Draper & Smith (3rd ed).
#
# Each "row" (i.e. each value of the first index) corresponds to a different
# reagent (labeled A, B, C and D in the text), and each "column" (i.e. each
# value of the second index) corresponds to a different catalyst (labeled 1,
# 2 and 3 in the text),
#
# For each reagent and catalyst, there are two production rates. So, for
# example, rates[2, 1] is [15, 9].  These two values are the production
# rates for the two tests conducted with reagent C and catalyst 2.

rates = np.array([[[ 4,  6], [11,  7], [ 5,  9]],
                  [[ 6,  4], [13, 15], [ 9,  7]],
                  [[13, 15], [15,  9], [13, 13]],
                  [[12, 12], [12, 14], [ 7,  9]]])

result = anova_twoway_balanced(rates)
print(result)
