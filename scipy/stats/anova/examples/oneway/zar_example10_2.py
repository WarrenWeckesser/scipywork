
import numpy as np
from anova import anova_oneway


t1 = [34, 36, 34, 35, 34]
t2 = [37, 36, 35, 37, 37]
t3 = [34, 37, 35, 37, 36]
t4 = [36, 34, 37, 34, 35]


result = anova_oneway(t1, t2, t3, t4)

with np.printoptions(formatter={'float': lambda t: format(t, '7.2f')}):
    print(result)
