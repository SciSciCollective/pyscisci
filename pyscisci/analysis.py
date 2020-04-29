# -*- coding: utf-8 -*-
"""
.. module:: analysis
    :synopsis: Set of functions for typical bibliometric analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import os
from functools import reduce
import pandas as pd
import numpy as np
from scipy import optimize


from .utils import isin_sorted, zip2dict, check4columns, fit_piecewise_linear, groupby_count, groupby_range


class CitationDataFrame(object):

    def __init__(self, pub2ref):
        self.pub2ref = pub2ref


    def disruption_index(self):
        return compute_disruption_index(self.pub2ref)


class AuthorCareerDataFrame(object):

    def __init__(self, author2pub):
        self.author2pub = author2pub


    def productivity(self, df=None, colgroupby = 'AuthorId', colcountby = 'PublicationId'):
        if df is None:
            df = self.author2pub

        newname_dict = zip2dict([str(colcountby), '0'], ['Productivity']*2)
        return groupby_count(df, colgroupby, colcountby, unique=True).rename(newname_dict)

    def yearly_productivity(self, df=None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId'):
         if df is None:
            df = self.author2pub

        newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
        return groupby_count(df, [colgroupby, datecol], colcountby, unique=True).rename(newname_dict)

    def career_length(self, df = None, colgroupby = 'AuthorId', colrange = 'Year'):
         if df is None:
            df = self.author2pub

        newname_dict = zip2dict([str(colrange)+'Range', '0'], ['CareerLength']*2)
        return groupby_range(df, colgroupby, colrange, unique=True).rename(newname_dict)

    def productivity_trajectory(self, df =None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId'):
        if df is None:
            df = self.author2pub

        return compute_yearly_productivity_traj(df, colgroupby = colgroupby)


    def hindex(self, df = None, colgroupby = 'AuthorId', colcountby = 'TotalCitations'):
        if df is None:
            df = self.author2pub
        return compute_hindex(df, colgroupby = colgroupby, colcountby = colcountby)



### H index

def article_hindex(a):
    d = np.sort(a)[::-1] - np.arange(a.shape[0])
    return (d>0).sum()

def compute_hindex(df, colgroupby, colcountby):
    """
    This function calculates the h index for the data frame.  See [h] for details.

    References
    ----------
    .. [h] Hirsh (2005): "title", *PNAS*.
           DOI: xxx
    """
    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'hindex']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].apply(article_hindex).to_frame().reset_index().rename(newname_dict)



### Productivity Trajectory

def fit_piecewise_lineardf(author_df):
    return fit_piecewise_linear(author_df['Year'].values, author_df['YearlyProductivity'].values)

def compute_yearly_productivity_traj(df, colgroupby = 'AuthorId'):
    """
    This function calculates the piecewise linear yearly productivity trajectory original studied in [w].

    References
    ----------
    .. [w] Way, Larremore (2018): "title", *PNAS*.
           DOI: xxx
    """

    newname_dict = zip2dict([str(i) for i in range(4)], ['t_break', 'b', 'm1', 'm2' ])
    return df.groupby(colgroupby, sort=False).apply(fit_piecewise_linear).to_frame().reset_index().rename(newname_dict)

### Disruption
def compute_disruption_index(pub2ref):
    """
    Li, Wang, Evans (2019) *Nature*
    """

    reference_groups = pub2ref.groupby('CitingPublicationId', sort = False)['CitedPublicationId']
    citation_groups = pub2ref.groupby('CitedPublicationId', sort = False)['CitingPublicationId']

    def get_citation_groups(pid):
        try:
            return citation_groups.get_group(pid).values
        except KeyError:
            return []

    def disruption_index(citing_focus):
        focusid = citing_focus.name

        # if the focus publication has no references, then it has a disruption of 1.0
        try:
            focusref = reference_groups.get_group(focusid)
        except KeyError:
            return 1.0

        cite2ref = reduce(np.union1d, [get_citation_groups(refid) for refid in focusref.values])

        nj = np.intersect1d(cite2ref, citing_focus.values).shape[0]
        ni = citing_focus.shape[0] - nj
        nk = cite2ref.shape[0] - nj
        return (ni - nj)/(ni + nj + nk)

    newname_dict = {'CitingPublicationId':'DisruptionIndex'}
    return citation_groups.apply(disruption_index).to_frame().reset_index().rename(columns = newname_dict)



### Novelty

def create_journalcitation_table(pubdf, pub2ref):
    required_pub_columns = ['PublicationId', 'JournalId', 'Year']
    check4columns(pubdf, required_pub_columns)
    pubdf = pubdf[required_pub_columns]

    required_pub2ref_columns = ['CitingPublicationId', 'CitedPublicationId']
    check4columns(pub2ref, required_pub_columns)
    pub2ref = pub2ref[required_pub2ref_columns]

    journals = np.sort(pubdf['JournalId'].unique())
    journal2int = {j:i for i,j in enumerate(journals)}
    pubdf['JournalInt'] = [journal2int[jid] for jid in pubdf['JournalId']]

    jctable = pub2ref.merge(pubdf[['PublicationId', 'Year', 'JournalInt']], how='left', left_on = 'CitingPublicationId', right_on = 'PublicationId')
    jctable.rename({'Year':'CitingYear', 'JournalInt':'CitingJournalInt'})
    del jctable['PublicationId']
    del jctable['CitingPublicationId']

    jctable = jctable.merge(pubdf[['PublicationId', 'Year', 'JournalInt']], how='left', left_on = 'CitedPublicationId', right_on = 'PublicationId')
    jctable.rename({'Year':'CitedYear', 'JournalInt':'CitedJournalInt'})
    del jctable['PublicationId']
    del jctable['CitedPublicationId']


    return jctable, {i:j for j,i in journal2int.items()}

def compute_novelty(pubdf, pub2ref, scratch_path = None, n_samples = 10):
    """
    This function calculates the novelty and conventionality for publications.

    References
    ----------
    .. [h] Uzzi, B. (2013): "title",
           *in submission*.
           DOI: xxx
    """

    journalcitation_table, int2journal = create_journalcitation_table(pubdf, pub2ref)

    for isample in range(n_samples):
        database_table = database_table.groupby(['CitingYear', 'CitedYear'], sort=False)['CitedJournalInt'].transform(np.random.permutation)

