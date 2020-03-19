import numpy as np
from anova import anova_oneway

# These example are from
# @book{Johnson,
#     author={Richard Johnson},
#     title={Miller \& Freund's Probability and Statistics for Engineers},
#     edition={seventh},
#     publisher={Pearson Prentice Hall},
#     address={Upper Saddle River, New Jersey},
#     year={2005}
# }


# Example on page 401

a = [13, 10, 8, 11, 8]
b = [13, 11, 14, 14]
c = [4, 1, 3, 4, 2, 4]
result = anova_oneway(a, b, c)

print("Example, page 401")
with np.printoptions(formatter={'float': lambda t: format(t, '8.3f')}):
    print(result)
print()


# Example on page 408

a = np.array([0.25, 0.27, 0.22, 0.30, 0.27, 0.28, 0.32, 0.24, 0.31, 0.26, 0.21, 0.28])
b = np.array([0.18, 0.28, 0.21, 0.23, 0.25, 0.20, 0.27, 0.19, 0.24, 0.22, 0.29, 0.16])
c = np.array([0.19, 0.25, 0.27, 0.24, 0.18, 0.26, 0.28, 0.24, 0.25, 0.20, 0.21, 0.19])
d = np.array([0.23, 0.30, 0.28, 0.28, 0.24, 0.34, 0.20, 0.18, 0.24, 0.28, 0.22, 0.21])

result = anova_oneway(a, b, c, d)
print("Example, page 408")
with np.printoptions(formatter={'float': lambda t: format(t, '8.5f')}):
    print(result)
print()


# Example on the bottom of page 409

pos1 = [90, 82, 79, 98, 83, 91]
pos2 = [105, 89, 93, 104, 89, 95, 86]
pos3 = [83, 89, 80, 94]

result = anova_oneway(pos1, pos2, pos3)

print("Example, page 409")
with np.printoptions(formatter={'float': lambda t: format(t, '8.3f')}):
    print(result)
