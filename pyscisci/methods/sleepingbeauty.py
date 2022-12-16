# -*- coding: utf-8 -*-
"""
.. module:: sleepingbeauty
    :synopsis: Calculate the sleeping beauty coefficient.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import sys

import pandas as pd
import numpy as np

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.utils import zip2dict


## Beauty-coefficient
def beauty_coefficient(c):
    """
    This function calculates the sleeping beauty coefficient and awakening time for a publication.  See :cite:`ke2015beauty` for details.

    Parameters
    ----------
    c : numpy array
        The yearly citation counts for the publication.

    Returns
    ----------
    B : float
        Sleeping Beauty Coefficient

    t_a : int
        The awakening time

    """
    c = c.values
    t_m = np.argmax(c)
    B_denom = c
    B_denom[c==0] = 1

    # :cite:`ke2015beauty` eq 1/2
    l_t = ((c[t_m] - c[0])/t_m *np.arange(c.shape[0]) + c[0] - c)/B_denom

    # :cite:`ke2015beauty` eq 2
    B = l_t[:(t_m+1)].sum()

    d_denom = np.sqrt((c[t_m] - c[0])**2 + t_m**2)
    d_t = np.abs( (c[t_m] - c[0]) * np.arange(c.shape[0]) + t_m * (c[0] - c)) / d_denom

    # :cite:`ke2015beauty` eq 3
    t_a = np.argmax(d_t[:(t_m+1)])

    return pd.Series([B, t_a], index=['BeautyCoefficient', 'Awakening'])

def compute_sleepingbeauty(df, colgroupby, colcountby, coldate='Year', show_progress=False):
    """
    Calculate the sleeping beauty and awakening time for each group in the DataFrame.  See :cite:`ke2015beauty` for details.

    The algorithmic implementation for each publication can be found in :py:func:`sleepingbeauty.beauty_coefficient`.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with the citation information for each publication in each year.

    colgroupby : str
        The DataFrame column with Publication Ids.

    colcountby : str
        The DataFrame column with Citation counts for each publication.

    coldate : str
        The DataFrame column with Year information.

    Returns
    -------
    DataFrame
        DataFrame with 3 columns: colgroupby, 'Beauty' and 'Awakening'

        """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Beauty', disable= not show_progress)

    def fill_missing_dates(subdf):
        subdf = subdf.set_index(coldate).reindex(np.arange(subdf[coldate].min(), subdf[coldate].max()+1)).fillna(0).reset_index()
        return subdf

    # first fill in missing dates
    df = df.groupby(colgroupby, sort=False, group_keys=False).apply(fill_missing_dates)

    #get start year
    syear = df.groupby(colgroupby, sort=False)[coldate].min()

    # now find the beauty coefficient and awakening year
    beauty = df.groupby(colgroupby, sort=False)[colcountby].progress_apply(beauty_coefficient).unstack(1).reset_index()

    # translate the awakening from index to year
    beauty['Awakening'] = [a+syear[pid] for pid,a in beauty[[colgroupby, 'Awakening']].values]
    
    return beauty