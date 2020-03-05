# ANOVA example from Section 23.6
# of Draper & Smith, Applied Regression Analysis, 3rd ed.
# The data is from Table 23.4, page 488.

import numpy as np
from anova import anova_twoway_balanced


rates = np.array(
    [[[4, 6], [6, 4], [13, 15], [12, 12]],
     [[11, 7], [13, 15], [15, 9], [12, 14]],
     [[5, 9], [9, 7], [13, 13], [7, 9]]])

result = anova_twoway_balanced(rates)
print(result)
