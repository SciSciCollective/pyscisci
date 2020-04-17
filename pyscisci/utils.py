# -*- coding: utf-8 -*-
"""
.. module:: publication
    :synopsis: The main Publication class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import numpy as np

def isin_range(values2check, min_value, max_value):
    return np.logical_and(values2check >= min_value, values2check <= max_value)

def isin_sorted(values2check, masterlist):
    index = np.searchsorted(masterlist, values2check, side = 'left')
    index[index >= masterlist.shape[0]] = masterlist.shape[0] - 1
    return values2check == masterlist[index]

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

