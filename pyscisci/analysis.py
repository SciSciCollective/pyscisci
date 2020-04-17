# -*- coding: utf-8 -*-
"""
.. module:: analysis
    :synopsis: Set of functions for typical bibliometric analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import os
import pandas as pd
import numpy as np

from utils import isin_sorted, zip2dict

def groupby_count(df, colgroupby, colcountby, unique=True):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Count']*2)
    if unique:
        return df.groupby(groupby, sort=False)[colcountby].nunique().to_frame().reset_index().rename(newname_dict)
    else:
        return df.groupby(groupby, sort=False)[colcountby].size().to_frame().reset_index().rename(newname_dict)

def groupby_range(df, colgroupby, colrange):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Range']*2)
    return df.groupby(colgroupby, sort=False)[colrange].apply(lambda x: x.max() - x.min()).to_frame().reset_index().rename(newname_dict)

def groupby_zero_col(df, colgroupby, colrange):
    return df.groupby(colgroupby, sort=False)[colrange].apply(lambda x: x - x.min()).to_frame().reset_index()

def groupby_total(df, colgroupby, coltotal):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colcountby)+'Total']*2)
    return df.groupby(colgroupby, sort=False)[colrange].sum().to_frame().reset_index().rename(newname_dict)

def productivity(df, colgroupby = 'AuthorId', colcountby = 'PublicationId'):
    newname_dict = zip2dict([str(colcountby), '0'], ['Productivity']*2)
    return groupby_count(df, colgroupby, colcountby, unique=True).rename(newname_dict)

def yearly_productivity(df, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId'):
    newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['AnnualProductivity']*2)
    return groupby_count(df, [colgroupby, datecol], colcountby, unique=True).rename(newname_dict)

def career_length(df, colgroupby = 'AuthorId', colrange = 'Year'):
    newname_dict = zip2dict([str(colrange)+'Range', '0'], ['CareerLength']*2)
    return groupby_range(df, colgroupby, colrange, unique=True).rename(newname_dict)

### H index

def hindex(a):
    d = np.sort(a)[::-1] - np.arange(a.shape[0])
    return (d>0).sum()

def compute_hindex(df, colgroupby, colcountby):
    newname_dict = zip2dict([str(colcountby), '0'], [str(colgroupby)+'hindex']*2)
    return df.groupby(colgroupby, sort=False)[colcountby].apply(hindex).to_frame().reset_index().rename(newname_dict)

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

