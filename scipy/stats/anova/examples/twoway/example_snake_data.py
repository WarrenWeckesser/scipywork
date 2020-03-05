import numpy as np
from anova import anova_twoway1


# From http://www.biostathandbook.com/twowayanova.html
# Snake-ID    Day-1   Day-2   Day-3   Day-4
# D1  85  58  15  57
# D3  107 51  30  12
# D5  61  60  68  36
# D8  22  41  63  21
# D11 40  45  28  10
# D12 65  27  3   16

data = np.array([[ 85, 58,  15,  57],
                 [107, 51,  30,  12],
                 [ 61, 60,  68,  36],
                 [ 22, 41,  63,  21],
                 [ 40, 45,  28,  10],
                 [ 65, 27,   3,  16]])

# The web page reports:
#   The effect of snake is not significant (F5, 15=1.24, P=0.34),
#   while the effect of day is significant (F3, 15=3.32, P=0.049).
# That agrees with the output of the following:
result = anova_twoway1(data)
print(result)
