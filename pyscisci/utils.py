# -*- coding: utf-8 -*-
"""
.. module:: utils
    :synopsis: utils for basic functions

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import sys
import pandas as pd
import numpy as np
import requests
import math
import scipy.stats as spstats

from numba import jit

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def groupby_count(df, colgroupby, colcountby, count_unique=True, show_progress=False):
    """
    Group the DataFrame and count the number for each group.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.

    colgroupby: str
        The column to groupby.

    colcountby: str
        The column to count.

    count_unique: bool, default True
        If True, count unique items in the rows.  If False, just return the number of rows.

    show_progress: bool or str, default False
        If True, display a progress bar for the count.  If str, the name of the progress bar to display.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colcountby+`Count`
    """
    
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Count']*2)

    if not show_progress:
        if count_unique:
            count = df.groupby(colgroupby, sort=False, as_index=False)[colcountby].nunique()
        else:
            count = df.groupby(colgroupby, sort=False, as_index=False)[colcountby].count()

    else:
        desc = ''
        if isinstance(show_progress, str):
            desc = show_progress
        # register our pandas apply with tqdm for a progress bar
        tqdm.pandas(desc=desc, disable= not show_progress)

        
        if count_unique:
            count = df.groupby(colgroupby, sort=False, as_index=False)[colcountby].progress_apply(lambda x: x.nunique())
        else:
            count = df.groupby(colgroupby, sort=False, as_index=False)[colcountby].progress_apply(lambda x: x.shape[0])

    return count.rename(columns=newname_dict)

def groupby_range(df, colgroupby, colrange, show_progress=False):
    """
   Group the DataFrame and find the range between the smallest and largest value for each group.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.

    colgroupby: str
        The column to groupby.

    colrange: str
        The column to find the range of values.

    show_progress: bool or str, default False
        If True, display a progress bar for the range.  If str, the name of the progress bar to display.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colrange+`Range`
    """
    desc = ''
    if isinstance(show_progress, str):
        desc = show_progress
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc=desc, disable= not show_progress)

    newname_dict = zip2dict([str(colrange), '0'], [str(colrange)+'Range']*2)
    return df.groupby(colgroupby, sort=False)[colrange].progress_apply(lambda x: x.max() - x.min()).to_frame().reset_index().rename(columns=newname_dict)

def groupby_zero_col(df, colgroupby, colrange, show_progress=False):
    """
    Group the DataFrame and shift the column so the minimum value is 0.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.

    colgroupby: str
        The column to groupby.

    colrange: str
        The column to find the range of values.

    show_progress: bool or str, default False
        If True, display a progress bar.  If str, the name of the progress bar to display.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colrange
    """
    desc = ''
    if isinstance(show_progress, str):
        desc = show_progress
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc=desc, disable= not show_progress)

    return df.groupby(colgroupby, sort=False)[colrange].progress_transform(lambda x: x - x.min())

def groupby_total(df, colgroupby, colcountby, show_progress=False):
    """
    Group the DataFrame and find the total of the column.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.

    colgroupby: str
        The column to groupby.

    colcountby: str
        The column to find the total of values.

    show_progress: bool or str, default False
        If True, display a progress bar for the summation.  If str, the name of the progress bar to display.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colcountby+'Total'
    """
    desc = ''
    if isinstance(show_progress, str):
        desc = show_progress
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc=desc, disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Total']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(lambda x: x.sum()).to_frame().reset_index().rename(columns=newname_dict)

def groupby_mean(df, colgroupby, colcountby, show_progress=False):
    """
    Group the DataFrame and find the mean of the column.

    Parameters
    ----------
    df: DataFrame
        The DataFrame.

    colgroupby: str
        The column to groupby.

    colcountby: str
        The column to find the mean of values.

    show_progress: bool or str, default False
        If True, display a progress bar for the summation.  If str, the name of the progress bar to display.

    Returns
    ----------
    DataFrame
        DataFrame with two columns: colgroupby, colcountby+'Mean'
    """
    desc = ''
    if isinstance(show_progress, str):
        desc = show_progress
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc=desc, disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Mean']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(lambda x: x.mean()).to_frame().reset_index().rename(columns=newname_dict)

def isin_range(values2check, min_value, max_value):
    """
    Check if the values2check are in the inclusive range [min_value, max_value].

    Parameters
    ----------
    values2check: numpy array
        The values to check.

    min_value: float
        The lowest value of the range.

    max_value: float
        The highest value of the range.

    show_progress: bool or str, default False
        If True, display a progress bar for the count.  If str, the name of the progress bar to display.

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
    values2check: numpy array
        The values to check.

    masterlist: numpy array
        The sorted list of master values.

    Returns
    ----------
    Numpy Array
        True if the value is in the masterlist.
    """
    index = np.searchsorted(masterlist, values2check, side = 'left')
    index[index >= masterlist.shape[0]] = masterlist.shape[0] - 1
    return values2check == masterlist[index]

def argtopk(a, k=5):
    return np.argpartition(a, -k)[-k:][::-1]

def changepoint(a):
    return np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [a.shape[0]]])

def holm_correction(pvalues, alpha=0.05):
    """
    Holm’s Step-Down Procedure for multiple hypothesis testing.
    Rather than controlling the FMER, Holm’s procedure controls for the false discovery rate (FDR)

    Parameters
    ----------
    pvalues: numpy array
        The p-values to check.

    alpha: float, defulat 0.05
        The significance level.

    Returns
    ----------
    Numpy Array
        True if the p-value is family-wise significant.

    """
    K = pvalues.shape[0]
    sorted_pvalues = np.sort(pvalues)
    holm_stepdown = alpha / (K - np.arange(K))
    
    first_reject_idx = np.argmax(sorted_pvalues > holm_stepdown)
    
    return np.argsort(pvalues) < first_reject_idx

def zip2dict(keys, values):
    return dict(zip(keys, values))

def series2df(ser):
    return ser.to_frame().reset_index()

def check4columns(df, column_list):
    # TODO: make proper error messages
    for col in column_list:
        if not col in list(df):
            raise KeyError("{} not in dataframe".format(col))

def pandas_cosine_similarity(df1, df2, col_keys, col_values):
    search_index = np.searchsorted(df2[col_keys].values, df1[col_keys].values, side = 'left')
    search_index[search_index >= df2.shape[0]] = df2.shape[0] - 1
    isin = df1[col_keys].values == df2[col_keys].values[search_index]
    cosine_num = np.inner(df1[col_values].values[isin], df2[col_values].values[search_index[isin]])
    cosine_denom = np.linalg.norm(df1[col_values].values) * np.linalg.norm(df2[col_values].values)
    return 1.0 - cosine_num/cosine_denom
    

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
    return -np.nansum(p[p>0]*np.log2(p[p>0]/q[p>0]))

def jenson_shannon(p,q):
    """
    Jensen–Shannon divergence
    """
    m = 0.5 * (p+q)
    return 0.5 * (kl(p,m) + kl(q,m))

def rank_array(a, ascending=True, normed=False):
    """
    Rank elements in the array.  
    ascending=> lowest=0, highest=1
    descending=> lowest=1, highest=0

    Parameters
    ----------
    a : numpy array or list
        Object to rank

    ascending : bool, default True
        Sort ascending vs. descending.

    normed : bool, default False
        False : rank is from 0 to N -1
        True : rank is from 0 to 1

    Returns
    ----------
    Ranked array.

    """
    idx = np.argsort(a)
    ranks = np.empty_like(idx)

    if ascending:
        ranks[idx] = np.arange(idx.shape[0])
    else:
        ranks[idx] = np.arange(idx.shape[0])[::-1]

    if normed:
        ranks = ranks/(ranks.shape[0]-1)
    return ranks

def convert_size(size_bytes):
    """
    Convert the bytes to human readable.
    """
    if size_bytes == 0:
       return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def empty_mode(a):
    """
    Mode of array when empty

    Parameters
    ----------
    a : numpy array or list
        array of values

    """
    values, counts = np.unique(a, return_counts=True)
    if values.shape[0] == 0:
        return None
    else:
        return np.random.choice(values[counts==counts.max()],1)[0]

@jit
def holder_mean(a, rho=1):
    """
    Holder mean

    Parameters
    ----------
    a : numpy array or list
        array of values

    rho : float
        holder parameter
        arithmetic mean (rho=1)
        geometric mean (rho=0)
        harmonic mean (rho=-1)
        quadratic mean (rho=2)
        max (rho-> infty)
        min (rho-> -infty)

    """
    return (a**rho).sum()**(1.0/rho) / a.shape[0]

def simpson(p):
    """
    Simpson diversity

    Parameters
    ----------
    p : numpy array or list
        array of values

    """
    un, counts = np.unique(p, return_counts=True)
    return np.square(counts/counts.sum()).sum()

def simpson_finite(p):
    """
    Simpson diversity with finite size correction

    Parameters
    ----------
    p : numpy array or list
        array of values

    """
    un, counts = np.unique(p, return_counts=True)
    N = counts.sum()
    if N > 1:
        return (counts * (counts - 1)).sum() / (N * (N - 1))
    else:
        return None

def shannon_entropy(p, base=np.e):
    """
    Shannon entropy

    Parameters
    ----------
    p : numpy array or list
        array of values

    """
    value,counts = np.unique(p, return_counts=True)
    return spstats.entropy(counts, base=base)

def value_to_int(a, sort_values='value', ascending=False, return_map=True):
    """
    Map the values of an array to integers.

    Parameters
    ----------
    a : numpy array or list
        array of values

    sort_values : str, default 'value'
        'none' : dont sort
        'value' : sort the array items based on their value
        'freq' : sort the array items based on their frequency (see ascending)

    ascending : bool, default False
        Only when sort_values == 'freq':
            False: larger counts dominate--map 0 to the most common
            True: smaller counts dominate--map 0 to the least common

    return_map: bool, default True
    """
    if sort_values == 'none':
        unique_values = pd.unique(a)
    
    elif sort_values == 'value':
        unique_values = np.sort(pd.unique(a))

    elif sort_values == 'freq':
        unique_values, freq = np.unique(a, return_counts=True)
        if ascending:
            unique_values = unique_values[np.argsort(freq)]
        else:
            unique_values = unique_values[np.argsort(freq)[::-1]]

    else:
        raise ValueError

    value_to_int_dict = {v:i for i,v in enumerate(unique_values)}
   
    if return_map:
        return [value_to_int_dict[v] for v in a], value_to_int_dict
    else:
        return [value_to_int_dict[v] for v in a]


def uniquemap_by_frequency(df, colgroupby='PublicationId', colcountby='FieldId', ascending=False):
    """
    Reduce a one-to-many mapping to a selection based on frequency of occurence in the dataframe  
    (either to the largest, most common or smallest, least common).
    
    Parameters
    ----------
    ascending : bool, default False
        False: larger counts dominate--map defaults to the most common
        True: smaller counts dominate--map defaults to the least common
    """
    countkeydict = {countid:i for i, countid in enumerate(df[colcountby].value_counts(ascending=ascending).index.values)}
    def countkey(a):
        if isinstance(a, pd.Series):
            return [countkeydict[c] for c in a]
        else:
            return countkeydict[a]
    return df.sort_values(by=[colcountby], key=countkey).drop_duplicates(subset=[colgroupby], keep='first')

@jit
def welford_mean_m2(previous_count, previous_mean, previous_m2, new_value):
    """
    Welford's algorithm for online mean and variance.
    """
    new_count = previous_count + 1
    new_mean = previous_mean + (new_value - previous_mean) / new_count
    new_m2 = previous_m2 + (new_value - previous_mean)*(new_value - new_mean)
    return (new_count, new_mean, new_m2)

@jit
def zscore_var(obs, m, var):
    return (obs-m) / np.sqrt(var)

@jit
def fast_delong(X,Y):
    """
    A fast algorithm from :cite:`Sun2014fastauc` to compute the DeLong AUC :cite:`DeLong1988auc`.
    """
    m = X.shape[0]
    n = Y.shape[0]

    if m == 0 or n == 0:
        return np.nan, np.nan

    theta = 0
    Z = np.concatenate((X,Y), axis = 0)
    Z_MR = compute_midrank(Z)
    X_MR = compute_midrank(X)
    Y_MR = compute_midrank(Y)


    auc = Z_MR[:m].sum() / m / n - float(m + 1.0) / 2.0 / n

    v01 = (Z_MR[:m] - X_MR) / n
    v10 = 1.0 - (Z_MR[m:] - Y_MR[:]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return auc, delongcov

@jit
def compute_midrank(x):
    """Computes midranks from :cite:`Sun2014fastauc`.

    Args:
       x - a 1D numpy array
    Returns:
       array of midranks

    """
    J = np.argsort(x)
    Z = x[J]
    N = x.shape[0]
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    
    T2[J] = T + 1
    return T2

def download_file_from_google_drive(file_id, destination=None, CHUNK_SIZE = 32768):
    """
    Download data files from the google Drive.

    Modified from: from https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    """

    raise NotImplementedError("Google Drive changed their download policy and this function no longer works.  A fix will come.")

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    if destination is None:
        return response
    else:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
        return None   

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None