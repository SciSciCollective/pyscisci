# -*- coding: utf-8 -*-
"""
.. module:: publication
    :synopsis: The main Publication class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import pandas as pd
import numpy as np
from scipy import optimize

def groupby_count(df, colgroupby, colcountby, unique=True):
    """
    Group the DataFrame and count the number for each group.

    Parameters
    ----------
    :param df: DataFrame
        The DataFrame.

    :param colgroupby: str
        The column to groupby.

    :param colcountby: str
        The column to count.

    :param unique: bool, default True
        If True, count unique items in the rows.  If False, just return the number of rows.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colcountby+`Count`
    """
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Count']*2)
    if unique:
        return df.groupby(colgroupby, sort=False)[colcountby].nunique().to_frame().reset_index().rename(columns=newname_dict)
    else:
        return df.groupby(colgroupby, sort=False)[colcountby].size().to_frame().reset_index().rename(columns=newname_dict)

def groupby_range(df, colgroupby, colrange):
    """
   Group the DataFrame and find the range between the smallest and largest value for each group.

    Parameters
    ----------
    :param df: DataFrame
        The DataFrame.

    :param colgroupby: str
        The column to groupby.

    :param colrange: str
        The column to find the range of values.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colrange+`Range`
    """
    newname_dict = zip2dict([str(colrange), '0'], [str(colrange)+'Range']*2)
    return df.groupby(colgroupby, sort=False)[colrange].apply(lambda x: x.max() - x.min()).to_frame().reset_index().rename(columns=newname_dict)

def groupby_zero_col(df, colgroupby, colrange):
    """
    Group the DataFrame and shift the column so the minimum value is 0.

    Parameters
    ----------
    :param df: DataFrame
        The DataFrame.

    :param colgroupby: str
        The column to groupby.

    :param colrange: str
        The column to find the range of values.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colrange
    """
    return df.groupby(colgroupby, sort=False)[colrange].transform(lambda x: x - x.min())

def groupby_total(df, colgroupby, colcountby):
    """
    Group the DataFrame and find the total of the column.

    Parameters
    ----------
    :param df: DataFrame
        The DataFrame.

    :param colgroupby: str
        The column to groupby.

    :param colcountby: str
        The column to find the total of values.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colcountby+'Total'
    """
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Total']*2)
    return df.groupby(colgroupby, sort=False)[colrange].sum().to_frame().reset_index().rename(columns=newname_dict)

def isin_range(values2check, min_value, max_value):
    """
    Check if the values2check are in the inclusive range [min_value, max_value].

    Parameters
    ----------
    :param values2check: numpy array
        The values to check.

    :param min_value: float
        The lowest value of the range.

    :param max_value: float
        The highest value of the range.

    Returns
    ----------
    Numpy Array
        True if the value is in the range.
    """
    return np.logical_and(values2check >= min_value, values2check <= max_value)

def isin_sorted(values2check, masterlist):
    """
    Check if the values2check are in the sorted masterlist.

    Parameters
    ----------
    :param values2check: numpy array
        The values to check.

    :param masterlist: numpy array
        The sorted list of master values.

    Returns
    ----------
    Numpy Array
        True if the value is in the masterlist.
    """
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
    return pd.Series(p)


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
