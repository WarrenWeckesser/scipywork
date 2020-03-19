import numpy as np
from anova import anova_oneway


f1 = [60.8, 67.0, 65.0, 68.6, 61.7]
f2 = [68.7, 67.7, 75.0, 73.3, 71.8]
f3 = [69.6, 77.1, 75.2, 71.5]
f4 = [61.9, 64.2, 63.1, 66.7, 60.3]

result = anova_oneway(f1, f2, f3, f4)

with np.printoptions(formatter={'float': lambda t: format(t, '8.3f')}):
    print(result)
