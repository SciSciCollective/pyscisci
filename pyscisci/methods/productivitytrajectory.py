# -*- coding: utf-8 -*-
"""
.. module:: productivitytrajectory
    :synopsis: Calculate the productivity trajectory.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import sys

import pandas as pd
import numpy as np

import scipy.optimize as spopt

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.utils import zip2dict


### Productivity Trajectory

def piecewise_linear(x, x_break, b, m1, m2):
    """
    A piece wise linear function:
    x <= x_break: y = m1*x + b
    x > x_break : y = m1*x_break + b + m2*(x - x_break)
    """
    return np.piecewise(x, [x <= x_break], [lambda x:m1*x + b-m1*x_break, lambda x:m2*x + b-m2*x_break])

def fit_piecewise_linear(xvalues, yvalues):
    m0 = (yvalues.max()-yvalues.min())/(xvalues.max() - xvalues.min())
    p0 = [np.median(xvalues), yvalues.mean(), m0, m0]
    p , e = spopt.curve_fit(piecewise_linear, xvalues, yvalues, p0=p0)
    return pd.Series(p)


def _fit_piecewise_lineardf(author, args):
    return fit_piecewise_linear(author[args[0]].values, author[args[1]].values)

def yearly_productivity_traj(df, colgroupby = 'AuthorId', colx='Year',coly='YearlyProductivity'):
    """
    Calculate the piecewise linear yearly productivity trajectory original studied in :cite:`way2017misleading`.

    """

    newname_dict = zip2dict(list(range(4)), ['t_break', 'b', 'm1', 'm2' ]) #[str(i) for i in range(4)]
    return df.groupby(colgroupby, sort=False).apply(_fit_piecewise_lineardf, args=(colx,coly) ).reset_index().rename(columns = newname_dict)

