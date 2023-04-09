import numpy as np
import math


# ' Goodman-Kruskal tau (scale is like multiple R^2, 
# '    take sqrt to match correlation scale)
# ' Reference: Code file provided by Professor Joe
# ' 
# ' @param ntab: two-way table of proportions or counts response variable in column
# ' @param ipror: If the value in the table is a proportion
# '
# ' @return dependent measure for row nominal variable to predict column
def GKtau(ntab, iprop=True):
    if not iprop:
        nn = ntab.values.sum()
        pp = ntab / nn
    else:
        pp = ntab
    px = np.sum(pp, axis=1)
    py = np.sum(pp, axis=0)
    vy = 1 - np.sum(py**2)
    xyTerm = np.apply_along_axis(lambda x: np.sum(x**2), 1, pp)
    vyBarx = 1 - np.sum(xyTerm / px)
    tauxPredy = 1 - vyBarx / vy
    return tauxPredy
