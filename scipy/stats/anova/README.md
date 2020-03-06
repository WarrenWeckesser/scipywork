ANOVA - Analysis of Variance
============================

This package, `anova`, is experimental and subject to change.
These notes describe the API of this package.  For a tutorial
on ANOVA, consult any undergraduate statistics text book, or
search online for tutorials, videos, etc.

One-way ANOVA
-------------
In one-way ANOVA, we have several groups of measurements.  Each
group corresponds to a set of measures made of some experimental
value for a given level of some "factor".  The question addressed
by one-way ANOVA is whether or not the means for each group are
all the same.  Generally, one-way ANOVA is used when there are
three or more groups.  With two groups, one would use the t test.

One-way ANOVA Example
---------------------
This example is from the wikipedia page

    https://en.wikipedia.org/wiki/One-way_analysis_of_variance#Example

Consider an experiment to study the effect of three different
levels of a factor on a response (e.g. three levels of a fertilizer
on plant growth). If we had 6 observations for each level, we could
write the outcome of the experiment in a table like this, where a1,
a2, and a3 are the three levels of the factor being studied.

    a1  a2  a3
    6   8   13
    8   12  9
    4   9   11
    5   11  8
    3   6   7
    4   8   12 

The null hypothesis, denoted H0, for the overall F-test for this
experiment would be that all three levels of the factor produce the
same response, on average.

Here's how we can use `anova.anova_oneway` to conduct the one-way
ANOVA test for this data.  We'll work in an ipython session:

```
In [1]: from anova import anova_oneway                                          

In [2]: a1 = [6, 8, 4, 5, 3, 4]                                                 

In [3]: a2 = [8, 12, 9, 11, 6, 8]                                               

In [4]: a3 = [13, 9, 11, 8, 7, 12]                                              

In [5]: result = anova_oneway(a1, a2, a3)                                       

In [6]: print(result)                                                           
ANOVA one-way
Source                   SS  DF          MS        F       p
Between groups     84.00000   2      42.000    9.265 0.0023988 
Within groups      68.00000  15       4.533
Total             152.00000  17       8.941
```
The F-statistic is 9.265, which corresponds to a p-value of 0.0024.
Our conclusion is that it is unlikely that the means of these three
groups are all the same.

Often the data that we went to analyze is not presented as
separate groups.  For example, the data given above might be
presented in two columns.  The first column indicates the
level, and the second indicates the value:

    Level  Value
        1      6
        2      8
        2     12
        3     13
        3      9
        2      9
        2     11
        1      8
        3     11
        1      4
        1      5
        1      3
        3      8
        1      4
        2      6
        2      8
        3      7
        3     12

This is how the data might be stored in, say, a text file.
To use `anova_oneway`, the data must be split into separate
arrays based on the first column.

Alternatively, one can use `anova_oneway_from_xy`.  This
function takes care of splitting the data into groups based
on the unique values found in the `x` array.  Suppose that
the data is stored in the file "levels.txt" in the two-column
format shown above.  Then we can write

```
In [1]: import numpy as np

In [2]: levels, values = np.loadtxt('levels.txt', skiprows=1, unpack=True, dtype=int)

In [3]: levels
Out[3]: array([1, 2, 2, 3, 3, 2, 2, 1, 3, 1, 1, 1, 3, 1, 2, 2, 3, 3])

In [4]: values
Out[4]: 
array([ 6,  8, 12, 13,  9,  9, 11,  8, 11,  4,  5,  3,  8,  4,  6,  8,  7,
       12])

In [5]: from anova import anova_oneway_from_xy

In [6]: result = anova_oneway_from_xy(levels, values)

In [7]: print(result)
ANOVA one-way
Source                   SS  DF          MS        F       p
Between groups     84.00000   2      42.000    9.265 0.0023988 
Within groups      68.00000  15       4.533
Total             152.00000  17       8.941
```

As expected, the results are the same as those returned by
`anova_oneway`.
