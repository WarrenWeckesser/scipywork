# ANOVA example from Sokal & Rohlf (fourth ed.), Box 11.1

import numpy as np
from anova import anova_twoway_balanced


consumption = np.array(
    [[[709, 679, 699],
      [592, 538, 476]],
     [[657, 594, 677],
      [508, 505, 539]]])

result = anova_twoway_balanced(consumption)
print(result)
