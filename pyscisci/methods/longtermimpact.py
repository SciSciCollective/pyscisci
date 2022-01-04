# -*- coding: utf-8 -*-
"""
.. module:: longterm impact
    :synopsis: Set of functions for typical bibliometric citation analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

import scipy.optimize as spopt
import scipy.stats as spstats

from pyscisci.utils import zip2dict


def _fit_predicted_citations(publication_citations):

    recenter_time = np.sort(publication_citations['DateDelta'].values)

    def fit_f(x):
        return np.arange(1,len(recenter_time) + 1) - np.array([predicted_c(t, x[0], x[1], x[2]) for t in recenter_time])

    s, _ = spopt.leastsq(fit_f, x0 = np.ones(3))
    return pd.Series(s)

def predicted_c(t, lam, mu, sig, m = 30.):
    lognormt = (np.log(t) - mu) / sig
    return m * (np.exp(lam * spstats.norm.cdf( lognormt ) ) - 1.0)

def longterm_impact(pub2ref, colgroupby = 'CitedPublicationId', coldate='CitingYear', show_progress=True):
    """
    This function calculates the longterm scientific impact as introduced in :cite:`Wang2013longterm`.

    Following equation (3) from [w]:
    c_(t) = m * (e^{lam * PHI()})

    """
    pub2ref = pub2ref.copy()

    if 'Year' in coldate:
        pub2ref['DateDelta'] = pub2ref.groupby(colgroupby, sort=False)[coldate].transform(lambda x: x-x.min())
    elif 'Date' in coldate:
        pub2ref['DateDelta'] = pub2ref.groupby(colgroupby, sort=False)[coldate].transform(lambda x: x-x.min()) / np.timedelta64(1,'D')
    else:
        print("Column Date Error") 

    pub2ref = pub2ref.loc[pub2ref['DateDelta'] > 0]
    pub2ref.sort_values(by=['DateDelta'], inplace=True)

    newname_dict = zip2dict(list(range(4)), ['lam', 'mu', 'sig', 'm' ])
    return pub2ref.groupby(colgroupby, sort=False).apply(_fit_predicted_citations).reset_index().rename(columns = newname_dict)
