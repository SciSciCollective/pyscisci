# -*- coding: utf-8 -*-
"""
.. module:: publicationmetrics
    :synopsis: Set of functions for the bibliometric analysis of publications

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

from pyscisci.utils import rank_array, check4columns

from pyscisci.methods.raostirling import *
from pyscisci.methods.diversity import *
from pyscisci.methods.creditshare import *
from pyscisci.methods.disruption import *
from pyscisci.methods.longtermimpact import *
from pyscisci.methods.sleepingbeauty import *
from pyscisci.methods.pivotscore import *
from pyscisci.methods.novelty import *


def citation_rank(df, colgroupby='Year', colrankby='C10', ascending=True, normed=False, show_progress=False):
    """
    Rank publications by the number of citations (smallest) to N -1 (largest)

    Parameters
    ----------
    df : DataFrame
        A DataFrame with the citation information for each Publication.

    colgroupby : str, list
        The DataFrame column(s) to subset by.

    colrankby : str
        The DataFrame column to rank by.

    ascending : bool, default True
        Sort ascending vs. descending.

    normed : bool, default False
        - False : rank is from 0 to N -1
        - True : rank is from 0 to 1

    show_progress : bool, default False
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

def publication_beauty(pub2ref , colgroupby = 'CitedPublicationId', colcountby = 'CitingPublicationId', show_progress=False):
    """
    Calculate the sleeping beauty and awakening time for each cited publication.  See :cite:`Sinatra2016qfactor` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.qfactor`.

    Parameters
    ----------
    pub2ref  : DataFrame, default None, Optional
        A DataFrame with the temporal citing information information.

    colgroupby : str, default 'CitedPublicationId', Optional
        The DataFrame column with Author Ids.  If None then the database 'CitedPublicationId' is used.

    colcountby : str, default 'CitingPublicationId', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'CitingPublicationId' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    check4columns(pub2ref , ['CitedPublicationId', 'CitingPublicationId', 'CitingYear'])

    tqdm.pandas(desc='Beauty', disable= not show_progress)

    df = groupby_count(pub2ref , colgroupby = ['CitedPublicationId', 'CitingYear'], colcountby = 'CitingPublicationId', count_unique = True)

    newname_dict = zip2dict([str(colcountby), '0', '1'], [str(colgroupby)+'Beauty']*2 + ['Awakening'])
    return df.groupby(colgroupby)[colcountby + 'Count'].progress_transform(beauty_coefficient).rename(columns=newname_dict)
