from statistics import mean, stdev

import scipy.stats as st
from math import sqrt


# Returns the 95% confidence interval, as seen in the R script from Patrício et al., 2018.
def ci(values):
    length = len(values)
    return (mean(values)) - st.norm.ppf(0.975) * sqrt(stdev(values) / length), \
           (mean(values)) + st.norm.ppf(0.975) * sqrt(stdev(values) / length)
