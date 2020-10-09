# -*- coding: utf-8 -*-
"""
.. module:: citationanalysis
    :synopsis: Set of functions for typical bibliometric citation analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import os
import sys
import itertools
from functools import reduce
from collections import defaultdict
import pandas as pd
import numpy as np
import scipy.sparse as spsparse
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.utils import isin_sorted, zip2dict, check4columns, fit_piecewise_linear, groupby_count, groupby_range, rank_array
from pyscisci.network import dataframe2bipartite, project_bipartite_mat, cocitation_network

def author_productivity(pub2author_df, colgroupby = 'AuthorId', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the total number of publications for each author.

    Parameters
    ----------
    :param pub2author_df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    :param colgroupby : str, default 'AuthorId', Optional
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
        show_progress='Author Productivity'

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['Productivity']*2)
    return groupby_count(pub2author_df, colgroupby, colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def author_yearly_productivity(pub2author_df, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the number of publications for each author in each year.

    Parameters
    ----------
    :param pub2author_df : DataFrame, default None, Optional
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
        show_progress='Yearly Productivity'

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
    return groupby_count(pub2author_df, [colgroupby, datecol], colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def author_career_length(pub2author_df = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the career length for each author.  The career length is the length of time from the first
    publication to the last publication.

    Parameters
    ----------
    :param pub2author_df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    :param colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param datecol : str, default 'Year', Optional
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'CareerLength'

    """

    # we can use show_progress to pass a label for the progress bar
    if show_progress:
        show_progress='Career Length'

    newname_dict = zip2dict([str(datecol)+'Range', '0'], ['CareerLength']*2)
    return groupby_range(pub2author_df, colgroupby, datecol, show_progress=show_progress).rename(columns=newname_dict)

def author_startyear(pub2author_df = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the year of first publication for each author.

    Parameters
    ----------
    :param pub2author_df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    :param colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param datecol : str, default 'Year', Optional
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'CareerLength'

    """

    newname_dict = zip2dict([str(datecol), '0'], ['StartYear']*2)
    return pub2author_df.groupby(colgroupby)[datecol].min().to_frame().reset_index().rename(columns=newname_dict)

def author_endyear(pub2author_df = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the year of last publication for each author.

    Parameters
    ----------
    :param pub2author_df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    :param colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param datecol : str, default 'Year', Optional
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'CareerLength'

    """

    newname_dict = zip2dict([str(datecol), '0'], ['EndYear']*2)
    return pub2author_df.groupby(colgroupby)[datecol].max().to_frame().reset_index().rename(columns=newname_dict)


def author_productivity_trajectory(pub2author_df, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`way2017misleading`

    The algorithmic implementation can be found in :py:func:`metrics.compute_yearly_productivity_traj`.

    Parameters
    ----------
    :param pub2author_df : DataFrame, default None
        A DataFrame with the author2publication information.

    :param colgroupby : str, default 'AuthorId'
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param datecol : str, default 'Year'
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    :param colcountby : str, default 'PublicationId'
        The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 5 columns: 'AuthorId', 't_break', 'b', 'm1', 'm2'

    """

    return yearly_productivity_traj(pub2author_df, colgroupby = colgroupby)

def author_hindex(pub2author_df, impact_df, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`hirsch2005index` for the derivation.

    The algorithmic implementation can be found in :py:func:`citationanalysis.compute_hindex`.

    Parameters
    ----------
    :param df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

    :param colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    :param colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    #df =

    if show_progress: print("Computing H-index.")
    return compute_hindex(pub2author_df.merge(impact_df[[colgroupby, colcountby]], on='PublicationId', how='left'),
     colgroupby = colgroupby,
     colcountby = colcountby,
     show_progress=show_progress)

## Q-factor
def qfactor(show_progress=False):
    """
    This function calculates the Q-factor for an author.  See [q] for details.

    References
    ----------
    .. [q] Sinatra (2016): "title", *Science*.
           DOI: xxx
    """

    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Q-factor', disable= not show_progress)

    # TODO: implement
    return False


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


### Productivity Trajectory

def _fit_piecewise_lineardf(author_df, args):
    return fit_piecewise_linear(author_df[args[0]].values, author_df[args[1]].values)

def yearly_productivity_traj(df, colgroupby = 'AuthorId', colx='Year',coly='YearlyProductivity'):
    """
    This function calculates the piecewise linear yearly productivity trajectory original studied in [w].

    References
    ----------
    .. [w] Way, Larremore (2018): "title", *PNAS*.
           DOI: xxx
    """

    newname_dict = zip2dict(list(range(4)), ['t_break', 'b', 'm1', 'm2' ]) #[str(i) for i in range(4)]
    return df.groupby(colgroupby, sort=False).apply(_fit_piecewise_lineardf, args=(colx,coly) ).reset_index().rename(columns = newname_dict)


class pySciSciMetricError(Exception):
    """
    Base Class for metric errors.
    """
    def __str__(self, msg=None):
        if msg is None:
            return 'pySciSci metric error.'
        else:
            return msg
