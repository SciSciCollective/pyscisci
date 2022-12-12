# -*- coding: utf-8 -*-
"""
.. module:: hindex
    :synopsis: Calculate the hindex.

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


### H index

def hindex(a):
    """
    Calculate the h index for the array of citation values.  See :cite:`hirsch2005index` for the definition.

    Parameters
    ----------
    :param a : numpy array
        An array of citation counts for each publication by the Author.

    Returns
    -------
    int
        The Hindex

    """
    d = np.sort(a)[::-1] - np.arange(a.shape[0])
    return (d>0).sum()

def compute_hindex(df, colgroupby, colcountby, show_progress=False):
    """
    Calculate the h index for each group in the DataFrame.  See :cite:`hirsch2005index` for the definition.

    The algorithmic implementation for each author can be found in :py:func:`citationanalysis.author_hindex`.

    Parameters
    ----------
    :param df : DataFrame
        A DataFrame with the citation information for each Author.

    :param colgroupby : str
        The DataFrame column with Author Ids.

    :param colcountby : str
        The DataFrame column with Citation counts for each publication.

    Returns
    -------
    DataFrame
        DataFrame with 2 columns: colgroupby, 'Hindex'

        """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Hindex', disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'Hindex']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(hindex).to_frame().reset_index().rename(columns=newname_dict)

def gindex(a):
    """
    Calculate the g index for the array of citation values.  See :cite:`Waltman2008index` for detailed definition.

    Parameters
    ----------
    :param a : numpy array
        An array of citation counts for each publication by the Author.

    Returns
    -------
    int
        The Gindex

    """
    d = np.cumsum(np.sort(a)[::-1]) - np.arange(a.shape[0])**2
    return (d>0).sum()

def compute_gindex(df, colgroupby, colcountby, show_progress=False):
    """
    Calculate the g index for each group in the DataFrame.  See :cite:`Waltman2008index` for detailed definition.

    The algorithmic implementation for each author can be found in :py:func:`citationanalysis.author_gindex`.

    Parameters
    ----------
    :param df : DataFrame
        A DataFrame with the citation information for each Author.

    :param colgroupby : str
        The DataFrame column with Author Ids.

    :param colcountby : str
        The DataFrame column with Citation counts for each publication.

    Returns
    -------
    DataFrame
        DataFrame with 2 columns: colgroupby, 'Hindex'

        """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Gindex', disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'Gindex']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(gindex).to_frame().reset_index().rename(columns=newname_dict)
