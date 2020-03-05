# ANOVA example from page 454 of Richard Johnson "Miller & Freund's Probability
# and Statistics for Engineers"

import numpy as np
from anova import anova_twoway_balanced


# XXX Check this against Johnson's result.

data = np.array(
    [[[3.5, 3.0, 2.7],
      [2.2, 2.3, 2.4]],
     [[7.1, 6.9, 7.5],
      [5.2, 4.6, 6.8]],
     [[10.8, 10.6, 11.0],
      [7.6, 7.1, 7.3]]])

result = anova_twoway_balanced(data)
print(result)
