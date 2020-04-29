# -*- coding: utf-8 -*-
"""
.. module:: publication
    :synopsis: The main Publication class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import pandas as pd
import numpy as np

def groupby_count(df, colgroupby, colcountby, unique=True):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Count']*2)
    if unique:
        return df.groupby(groupby, sort=False)[colcountby].nunique().to_frame().reset_index().rename(newname_dict)
    else:
        return df.groupby(groupby, sort=False)[colcountby].size().to_frame().reset_index().rename(newname_dict)

def groupby_range(df, colgroupby, colrange):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Range']*2)
    return df.groupby(colgroupby, sort=False)[colrange].apply(lambda x: x.max() - x.min()).to_frame().reset_index().rename(newname_dict)

def groupby_zero_col(df, colgroupby, colrange):
    return df.groupby(colgroupby, sort=False)[colrange].apply(lambda x: x - x.min()).to_frame().reset_index()

def groupby_total(df, colgroupby, coltotal):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Total']*2)
    return df.groupby(colgroupby, sort=False)[colrange].sum().to_frame().reset_index().rename(newname_dict)

def isin_range(values2check, min_value, max_value):
    return np.logical_and(values2check >= min_value, values2check <= max_value)

def isin_sorted(values2check, masterlist):
    index = np.searchsorted(masterlist, values2check, side = 'left')
    index[index >= masterlist.shape[0]] = masterlist.shape[0] - 1
    return values2check == masterlist[index]

def piecewise_linear(x, x_break, b, m1, m2):
    """
    A piece wise linear function:
    x <= x_break: y = m1*x + b
    x > x_break : y = m1*x_break + b + m2*(x - x_break)
    """
    return np.piecewise(x, [x <= x_break], [lambda x:m1*x + b, lambda x:m1*x_break + b + m2*(x-x_break)])

def fit_piecewise_linear(xvalues, yvalues):
    p , e = optimize.curve_fit(piecewise_linear, xvalues, yvalues)
    return p


def changepoint(a):
    return np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [a.shape[0]]])

def zip2dict(keys, values):
    return dict(zip(keys, values))

def series2df(ser):
    return ser.to_frame().reset_index()

def load_int(v):
    try:
        return int(v)
    except ValueError:
        return None

def load_float(v):
    try:
        return float(v)
    except ValueError:
        return None

def check4columns(df, column_list):
    # TODO: make proper error messages
    for col in column_list:
        if not col in list(df):
            print("Must pass {0} in {1}".format(col, df.name))

def rolling_window(a, window, step_size = 1):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def forward_rolling_window(a, window, step_size = 1):
    a = np.pad(a.astype(float), (0,int(window)), 'constant', constant_values=(np.nan, np.nan))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def hard_rolling_window(a, window, step_size = 1):
    a = np.pad(a.astype(float), (int((window - 1)/2),int((window - 1)/2) + 1), 'constant', constant_values=(np.nan, np.nan))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return  np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def past_window(a, window, step_size = 1):
    a = np.pad(a.astype(float), (window - 1, 0), 'constant', constant_values=(np.nan, np.nan))
    shape = a.shape[:-1] + (a.shape[-1] - window + 2 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return  np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def kl(p,q):
    """
    Kullback–Leibler divergence (KL-divergence)
    """
    return np.nansum(p*np.log2(p/q))

def jenson_shannon(p,q):
    """
    Jensen–Shannon divergence
    """
    m = 0.5 * (p+q)
    return 0.5 * (kl(p,m) + kl(q,m))
