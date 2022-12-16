# -*- coding: utf-8 -*-
"""
.. module:: qfactor
    :synopsis: Calculate the qfactor.

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


## Q-factor
def qfactor(a):
    """
    This function calculates the Q-factor for an author.  See :cite:`Sinatra2016individual` for details.

    """

    return np.exp(np.mean(np.log(a[a>0])))

def compute_qfactor(df, colgroupby, colcountby, show_progress=False):
    """
    Calculate the q factor for each group in the DataFrame.  See :cite:`Sinatra2016individual` for the definition.

    The algorithmic implementation for each author can be found in :py:func:`citationanalysis.author_qfactor`.

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
    tqdm.pandas(desc='Qfactor', disable= not show_progress)

    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'Qfactor']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(qfactor).to_frame().reset_index().rename(columns=newname_dict)