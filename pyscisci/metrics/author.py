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

from pyscisci.utils import zip2dict, check4columns, groupby_count, groupby_range
from pyscisci.metrics.hindex import compute_hindex
from pyscisci.metrics.qfactor import compute_qfactor
from pyscisci.metrics.productivitytrajectory import yearly_productivity_traj
from pyscisci.metrics.diffusionscientificcredit import diffusion_of_scientific_credit

def author_productivity(pub2author_df, colgroupby = 'AuthorId', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the total number of publications for each author.

    Parameters
    ----------
    pub2author_df : DataFrame, default None, Optional
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

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['Productivity']*2)
    return groupby_count(pub2author_df, colgroupby, colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def author_yearly_productivity(pub2author_df, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the number of publications for each author in each year.

    Parameters
    ----------
    pub2author_df : DataFrame, default None, Optional
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

    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
    return groupby_count(pub2author_df, [colgroupby, datecol], colcountby, count_unique=True, show_progress=show_progress).rename(columns=newname_dict)

def author_career_length(pub2author_df = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the career length for each author.  The career length is the length of time from the first
    publication to the last publication.

    Parameters
    ----------
    pub2author_df : DataFrame, default None, Optional
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
    return groupby_range(pub2author_df, colgroupby, datecol, show_progress=show_progress).rename(columns=newname_dict)

def author_startyear(pub2author_df = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the year of first publication for each author.

    Parameters
    ----------
    pub2author_df : DataFrame, default None, Optional
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
    return pub2author_df.groupby(colgroupby)[datecol].min().to_frame().reset_index().rename(columns=newname_dict)

def author_endyear(pub2author_df = None, colgroupby = 'AuthorId', datecol = 'Year', show_progress=False):
    """
    Calculate the year of last publication for each author.

    Parameters
    ----------
    pub2author_df : DataFrame, default None, Optional
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
    return pub2author_df.groupby(colgroupby)[datecol].max().to_frame().reset_index().rename(columns=newname_dict)


def author_productivity_trajectory(pub2author_df, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:p:`way2017misleading`

    The algorithmic implementation can be found in :py:func:`metrics.compute_yearly_productivity_traj`.

    Parameters
    ----------
    pub2author_df : DataFrame, default None
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
    if not 'YearlyProductivity' in list(pub2author_df):
        yearlyprod = author_yearly_productivity(pub2author_df, colgroupby=colgroupby, datecol=datecol, colcountby=colcountby)
    else:
        yearlyprod = pub2author_df
    return yearly_productivity_traj(yearlyprod, colgroupby = colgroupby)

def author_hindex(pub2author_df, impact_df=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`hirsch2005index` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.hindex`.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

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
    if impact_df is None:
        df = pub2author_df
    else:
        df = pub2author_df.merge(impact_df[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_hindex(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_qfactor(pub2author_df, impact_df=None, colgroupby = 'AuthorId', colcountby = 'Ctotal', show_progress=False):
    """
    Calculate the author yearly productivity trajectory.  See :cite:`Sinatra2016qfactor` for the derivation.

    The algorithmic implementation can be found in :py:func:`metrics.qfactor`.

    Parameters
    ----------
    df : DataFrame, default None, Optional
        A DataFrame with the author2publication information.  If None then the database 'author2pub_df' is used.

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
    if impact_df is None:
        df = pub2author_df
    else:
        df = pub2author_df.merge(impact_df[[colgroupby, colcountby]], on='PublicationId', how='left')

    return compute_qfactor(df, colgroupby = colgroupby, colcountby = colcountby, show_progress=show_progress)

def author_top_field(pub2author_df, colgroupby = 'AuthorId', colcountby = 'FieldId', fractional_field_counts = False, show_progress=False):
    """
    Calculate the most frequent field in the authors career.

    Parameters
    ----------
    pub2author_df : DataFrame
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

    check4columns(pub2author_df, [colgroupby, 'PublicationId', colcountby])

    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Author Top Field', disable= not show_progress)

    if not fractional_field_counts:
        author2field = pub2author_df.groupby(colgroupby)[colcountby].progress_apply(lambda x: x.mode()[0])

    else:
        # first calculate how many fields each publication maps too
        pub2nfields = groupby_count(pub2author_df, colgroupby='PublicationId', colcountby=colcountby)

        # each pub2field mapping is weighted by the number of fields for the publication
        pub2nfields['PublicationWeight'] = 1.0/pub2nfields['PublicationIdCount']
        del pub2nfields[str(colcountby)+'Count']

        # merge counts
        author2field = pub2author_df.merge(pub2nfields, how='left', on='PublicationId')

        # custom weighted mode based on 
        def weighted_mode(adf):
            p = adf.groupby(colcountby)['PublicationWeight'].sum()
            return p.idxmax()

        # now take the weighted mode for each groupby column
        author2field = author2field.groupby(colgroupby).progress_apply(weighted_mode)

    newname_dict = zip2dict([str(colcountby), '0'], ['Top' + str(colcountby)]*2)
    return author2field.to_frame().reset_index().rename(columns=newname_dict)





class pySciSciMetricError(Exception):
    """
    Base Class for metric errors.
    """
    def __str__(self, msg=None):
        if msg is None:
            return 'pySciSci metric error.'
        else:
            return msg
