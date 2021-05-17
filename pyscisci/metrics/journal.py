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

from pyscisci.utils import zip2dict, groupby_count
from pyscisci.metrics.hindex import compute_hindex

def journal_productivity(pub2journal_df, colgroupby = 'JournalId', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the total number of publications for each journal.

    Parameters
    ----------
    :param pub2journal_df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    :param colgroupby : str, default 'JournalId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param colcountby : str, default 'PublicationId', Optional
        The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.


    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'Productivity'

    """

    # we can use show_progress to pass a label for the progress bar
    if show_progress:
        show_progress='Journal Productivity'

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['Productivity']*2)
    return groupby_count(pub2journal_df, colgroupby, colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def journal_yearly_productivity(pub2journal_df, colgroupby = 'JournalId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the number of publications for each author in each year.

    Parameters
    ----------
    :param pub2journal_df : DataFrame, default None, Optional
        A DataFrame with the author2publication information. 

    :param colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param datecol : str, default 'Year', Optional
        The DataFrame column with Year information.  If None then the database 'Year' is used.

    :param colcountby : str, default 'PublicationId', Optional
        The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 3 columns: 'AuthorId', 'Year', 'YearlyProductivity'

    """
    
    # we can use show_progress to pass a label for the progress bar
    if show_progress:
        show_progress='Journal Yearly Productivity'

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
    return groupby_count(pub2journal_df, [colgroupby, datecol], colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def journal_hindex(pub2journal_df, impact_df=None, colgroupby = 'JournalId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`hirsch2005index` for the derivation.

    The algorithmic implementation can be found in :py:func:`citationanalysis.compute_hindex`.

    Parameters
    ----------
    :param pub2journal_df : DataFrame
        A DataFrame with the publication and journal information.

    :param impact_df : DataFrame, default None, Optional
        A DataFrame with the publication citation counts precalculated.  If None, then it is assumed that the citation counts are already in pub2journal_df.

    :param colgroupby : str, default 'JournalId', Optional
        The DataFrame column with Author Ids.  If None then the database 'JournalId' is used.

    :param colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    :param show_progress : bool, default False
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.
    
    :param show_progress: bool, default False
            Show progress of the calculation.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'JournalId', 'Hindex'

    """
    if not impact_df is None:
        pub2journal_df = pub2journal_df.merge(impact_df[[colgroupby, colcountby]], on='PublicationId', how='left')

    if show_progress: print("Computing Journal H-index.")
    return compute_hindex(pub2journal_df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def journal_impactfactor(pub_df, pub2ref_df, pub2year=None, citation_window=5, colgroupby = 'JournalId', show_progress=False):
    """
    Calculate the impact factor for a journal.

    Parameters
    ----------
    :param pub_df : DataFrame
        A DataFrame with the publication, journal, and year information.

    :param pub2ref_df : DataFrame
        A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

    :param pub2year : dict, defualt None, Optional
        A dictionary mapping 'PublicationIds' to the publication year.  If None then the 'CitingYear' is assumed to be a column of pub2ref_df.

    :param colgroupby : str, default 'JournalId', Optional
        The DataFrame column with Author Ids.  If None then the database 'JournalId' is used.

    :param colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.
    
    :param show_progress: bool, default False
            Show progress of the calculation.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'JournalId', 'ImpactFactor{y}' where y is the citation_window size

    """

