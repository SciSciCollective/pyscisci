# -*- coding: utf-8 -*-
"""
.. module:: citationanalysis
    :synopsis: Set of functions for typical bibliometric citation analysis

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

from pyscisci.utils import rank_array

from pyscisci.metrics.raostirling import *
from pyscisci.metrics.creditshare import *
from pyscisci.metrics.disruption import *
from pyscisci.metrics.longtermimpact import *

def citation_rank(df, colgroupby='Year', colrankby='C10', ascending=True, normed=False, show_progress=False):
    """
    Rank elements in the array from 0 (smallest) to N -1 (largest)

    Parameters
    ----------
    :param df : DataFrame
        A DataFrame with the citation information for each Publication.

    :param colgroupby : str, list
        The DataFrame column(s) to subset by.

    :param colrankby : str
        The DataFrame column to rank by.

    :param ascending : bool, default True
        Sort ascending vs. descending.

    :param normed : bool, default False
        False : rank is from 0 to N -1
        True : rank is from 0 to 1

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    DataFrame
        The original dataframe with a new column for rank: colrankby+"Rank"

    """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Citation Rank', disable= not show_progress)

    df[str(colrankby)+"Rank"] = df.groupby(colgroupby)[colrankby].progress_transform(lambda x: rank_array(x, ascending, normed))
    return df
