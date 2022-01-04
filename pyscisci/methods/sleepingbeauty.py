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

    return np.Series([B, t_a])

def compute_sleepingbeauty(df, colgroupby, colcountby, show_progress=False):
    """
    Calculate the sleeping beauty and awakening time for each group in the DataFrame.  See :cite:`ke2015beauty` for details.

    The algorithmic implementation for each publication can be found in :py:func:`sleepingbeauty.beauty_coefficient`.

    Parameters
    ----------
    df : DataFrame
        A DataFrame with the citation information for each Author.

    colgroupby : str
        The DataFrame column with Author Ids.

    colcountby : str
        The DataFrame column with Citation counts for each publication.

    Returns
    -------
    DataFrame
        DataFrame with 3 columns: colgroupby, 'Beauty' and 'Awakening'

        """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Beauty', disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0', '1'], [str(colgroupby)+'Beauty']*2 + ['Awakening'])
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(beauty_coefficient).to_frame().reset_index().rename(columns=newname_dict)
