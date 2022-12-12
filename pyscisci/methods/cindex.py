# -*- coding: utf-8 -*-
"""
.. module:: cindex
    :synopsis: Calculate the cindex.

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


def compute_cindex(df, colgroupby, colcountby, show_progress=False):
    """
    Calculate the cindex for each group in the DataFrame (the number of citations to the maximum cited publication).
    See :cite:`Waltman2008index` for detailed definition.

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
        DataFrame with 2 columns: colgroupby, 'Cindex'

        """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='cindex', disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'Cindex']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].max(qfactor).to_frame().reset_index().rename(columns=newname_dict)

def compute_pindex(df, colgroupby, colcountby, show_progress=False):
    """
    Calculate the pindex for each group in the DataFrame (the number of publications with >0 citations).  
    See :cite:`Waltman2008index` for detailed definition.

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
        DataFrame with 2 columns: colgroupby, 'Pindex'

        """
    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='pindex', disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'Pindex']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].astype(bool).sum(axis=0).to_frame().reset_index().rename(columns=newname_dict)

