# -*- coding: utf-8 -*-
"""
.. module:: authormetrics
    :synopsis: Set of functions for the bibliometric analysis of authors

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

from pyscisci.utils import zip2dict, check4columns, groupby_count, groupby_range, empty_mode
from pyscisci.methods.hindex import compute_hindex, compute_gindex
from pyscisci.methods.qfactor import compute_qfactor
from pyscisci.methods.productivitytrajectory import yearly_productivity_traj
from pyscisci.methods.diffusionscientificcredit import diffusion_of_scientific_credit
from pyscisci.methods.careertopics import career_cociting_network_topics
from pyscisci.methods.hotstreak import career_hotstreak

def author_productivity(pub2author=None, colgroupby = 'AuthorId', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the total number of publications for each author.

    Parameters
    ----------
    pub2author : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'PublicationId', Optional
        The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.


    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'Productivity'

    """

    # we can use show_progress to pass a label for the progress bar
    if show_progress:
        show_progress='Author Productivity'

    if pub2author is None:
        pub2author = self.pub2author

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['Productivity']*2)
    return groupby_count(pub2author, colgroupby, colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def author_yearly_productivity(pub2author=None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the number of publications for each author in each year.

    Parameters
    ----------
    pub2author : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    datecol : str, default 'Year', Optional
        The DataFrame column with Year information.  If None then the database 'Year' is used.

    colcountby : str, default 'PublicationId', Optional
        The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 3 columns: 'AuthorId', 'Year', 'YearlyProductivity'

    """

    # we can use show_progress to pass a label for the progress bar
    if show_progress:
        show_progress='Yearly Productivity'

    if pub2author is None:
        pub2author = self.pub2author

    if not 'Year' in list(pub2author):
        pub2year = self.pub2year
        pub2author['Year'] = [pub2year.get(pid, None) for pid in pub2author['PublicationId'].values]
        pub2author.dropna(subset=['Year'], inplace=True)

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
    return groupby_count(pub2author, [colgroupby, datecol], colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def author_career_length(pub2author = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the career length for each author.  The career length is the length of time from the first
    publication to the last publication.

    Parameters
    ----------
    pub2author : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    datecol : str, default 'Year', Optional
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
    return groupby_range(pub2author, colgroupby, datecol, show_progress=show_progress).rename(columns=newname_dict)

def author_startyear(pub2author = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the year of first publication for each author.

    Parameters
    ----------
    pub2author : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    datecol : str, default 'Year', Optional
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'CareerLength'

    """

    newname_dict = zip2dict([str(datecol), '0'], ['StartYear']*2)
    return pub2author.groupby(colgroupby)[datecol].min().to_frame().reset_index().rename(columns=newname_dict)

def author_endyear(pub2author = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the year of last publication for each author.

    Parameters
    ----------
    pub2author : DataFrame, default None, Optional
        A DataFrame with the author2publication information.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    datecol : str, default 'Year', Optional
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    Returns
    -------
    DataFrame
        Productivity DataFrame with 2 columns: 'AuthorId', 'CareerLength'

    """

    newname_dict = zip2dict([str(datecol), '0'], ['EndYear']*2)
    return pub2author.groupby(colgroupby)[datecol].max().to_frame().reset_index().rename(columns=newname_dict)


def author_productivity_trajectory(pub2author, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:p:`way2017misleading`

    The algorithmic implementation can be found in :py:func:`metrics.compute_yearly_productivity_traj`.

    Parameters
    ----------
    pub2author : DataFrame, default None
        A DataFrame with the author2publication information.

    colgroupby : str, default 'AuthorId'
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    datecol : str, default 'Year'
        The DataFrame column with Date information.  If None then the database 'Year' is used.

    colcountby : str, default 'PublicationId'
        The DataFrame column with Publication Ids.  If None then the database 'PublicationId' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 5 columns: 'AuthorId', 't_break', 'b', 'm1', 'm2'

    """
    if not 'YearlyProductivity' in list(pub2author):
        yearlyprod = author_yearly_productivity(pub2author, colgroupby=colgroupby, datecol=datecol, colcountby=colcountby)
    else:
        yearlyprod = pub2author
    return yearly_productivity_traj(yearlyprod, colgroupby = colgroupby)

def author_hindex(pub2author, impact=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author H-index.  See :cite:`hirsch2005index` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.hindex`.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub' is used.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    #df =

    if show_progress: print("Computing H-index.")
    if impact is None:
        df = pub2author
    else:
        df = pub2author.merge(impact[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_hindex(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_gindex(pub2author, impact=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author g-index.  See :cite:`hirsch2005index` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.hindex`.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub' is used.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    #df =

    if show_progress: print("Computing G-index.")
    if impact is None:
        df = pub2author
    else:
        df = pub2author.merge(impact[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_gindex(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_hindex(pub2author, impact=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`hirsch2005index` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.hindex`.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub' is used.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    #df =

    if show_progress: print("Computing H-index.")
    if impact is None:
        df = pub2author
    else:
        df = pub2author.merge(impact[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_hindex(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_qfactor(pub2author, impact=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`Sinatra2016qfactor` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.qfactor`.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub' is used.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    #df =

    if show_progress: print("Computing Q-factor.")
    if impact is None:
        df = pub2author
    else:
        df = pub2author.merge(impact[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_qfactor(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_cindex(pub2author, impact=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author c-index.  See :cite:`Waltman2008index` for the derivation.

    The number of citations for an author's most cited work.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub' is used.

    colgroupby : str, default 'AuthorId', Optional
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'Ctotal', Optional
        The DataFrame column with Citation counts for each publication.  If None then the database 'Ctotal' is used.

    Returns
    -------
    DataFrame
        Trajectory DataFrame with 2 columns: 'AuthorId', 'Hindex'

    """

    #df =

    if show_progress: print("Computing c-index.")
    if impact is None:
        df = pub2author
    else:
        df = pub2author.merge(impact[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_qfactor(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_top_field(pub2author, colgroupby = 'AuthorId', colcountby = 'FieldId', 
    fractional_field_counts = False, show_progress=False):
    """
    Calculate the most frequent field in the authors career.

    Parameters
    ----------
    pub2author : DataFrame
        A DataFrame with the author2publication field information.

    colgroupby : str, default 'AuthorId'
        The DataFrame column with Author Ids.  If None then the database 'AuthorId' is used.

    colcountby : str, default 'FieldId'
        The DataFrame column with Citation counts for each publication.  If None then the database 'FieldId' is used.

    fractional_field_counts : bool, default False
        How to count publications that are assigned to multiple fields:
            - If False, each publication-field assignment is counted once.
            - If True, each publication is counted once, contributing 1/#fields to each field.

    Returns
    -------
    DataFrame
        DataFrame with 2 columns: 'AuthorId', 'TopFieldId'

    """

    check4columns(pub2author, [colgroupby, 'PublicationId', colcountby])

    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Author Top Field', disable= not show_progress)

    if not fractional_field_counts:
        author2field = pub2author.groupby(colgroupby)[colcountby].progress_apply(empty_mode)
    else:
        # first calculate how many fields each publication maps too
        pub2nfields = groupby_count(pub2author, colgroupby='PublicationId', colcountby=colcountby)

        # each pub2field mapping is weighted by the number of fields for the publication
        pub2nfields['PublicationWeight'] = 1.0/pub2nfields['PublicationIdCount']
        del pub2nfields[str(colcountby)+'Count']

        # merge counts
        author2field = pub2author.merge(pub2nfields, how='left', on='PublicationId')

        # custom weighted mode based on 
        def weighted_mode(adf):
            p = adf.groupby(colcountby)['PublicationWeight'].sum()
            return p.idxmax()

        # now take the weighted mode for each groupby column
        author2field = author2field.groupby(colgroupby).progress_apply(weighted_mode)

    newname_dict = zip2dict([str(colcountby), '0'], ['Top' + str(colcountby)]*2)
    return author2field.to_frame().reset_index().rename(columns=newname_dict)


def author_hotstreak(pub2author, colgroupby = 'AuthorId', citecol = 'c10',datecol='Year',  maxk=1, l1_lambda=1.0, show_progress=False):
    """
    Identify hot streaks in author careers :cite:`liu2018hotstreak'.

    TODO: this is an interger programming problem.  Reimplement using an interger solver.
    Right now just using a brut force search (very inefficient)!
    
    Parameters
    ----------
    pub2author : DataFrame
        The author publication history for all authors.

    colgroupby : str, default 'AuthorId'
        The column with Author information.

    citecol : str, default 'c10'
        The column with publication citation information.

    datecol : str, default 'Year'
        The column with publication date/year information.
    
    max_k : int, default 1
        The maximum number of hot streaks to search for in a career. Should be 1 or 2.
    
    l1_lambda : float, default 1.0
        The l1 regularization for the number of streaks.  
        Note, the authors never define the value they used for this in the SI.
        
    Returns
    ----------
    lsm_err : float
        The least square mean error of the model plus the l1-regularized term for the number of model coefficients.

    streak_loc : array
        The index locations for the hot streak start and end locations.
    """

    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Author HotStreak', disable= not show_progress)

    pub2author = pub2author.sort_values(by=[colgroupby, datecol]).reset_index(drop=True)

    hotstreak = pub2author.groupby(colgroupby).progress_apply(lambda x: career_hotstreak(x, citecol=citecol, maxk=maxk, l1_lambda = l1_lambda))
    return hotstreak.reset_index().rename(columns={'level_1':'StreakNumber'})

class pySciSciMetricError(Exception):
    """
    Base Class for metric errors.
    """
    def __str__(self, msg=None):
        if msg is None:
            return 'pySciSci metric error.'
        else:
            return msg
