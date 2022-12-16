# -*- coding: utf-8 -*-
"""
.. module:: reference strength
    :synopsis: Set of functions for finding field reference strength and share.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

from collections import defaultdict

import scipy.sparse as spsparse

from sklearn.preprocessing import normalize

from ..utils import isin_sorted, zip2dict, check4columns
from ..network import dataframe2bipartite


def field_citation_vectors(source2target, pub2field, count_normalize=None, n_fields = None, citation_direction='references'):
    """
    Calculate the citation vector of reference/citation counts from a field to another field.

    Parameters
    ----------
    :param pub2reffield : DataFrame
        A DataFrame with the citation relationships and fields for each Publication.

    count_normalize : int, default None
        If None: use the raw reference/citation counts.
        If 0: Normalize by the Target Counts
        If 1: Normalize by the Source Counts

    :param citation_direction : str, default `references`
        `references` : the fields are defined by a publication's references.
        `citations` : the fields are defined by a publication's citations.

    Returns
    -------
    Distance DataFrame
        DataFrame with 3 columns: SourceFieldId, TargetFieldId, Count

    """

    if n_fields is None:
        Nfields = int(pub2field['FieldId'].max()) + 1
    else:
        Nfields = n_fields

    source2target = source2target.merge(pub2field[['PublicationId', 'FieldId', 'PubFieldContribution']], how='left', left_on='SourceId', right_on='PublicationId').rename(
            columns={'FieldId':'SourceFieldId', 'PubFieldContribution':'SourcePubFieldContribution'})
    del source2target['PublicationId']

    source2target = source2target.merge(pub2field[['PublicationId', 'FieldId', 'PubFieldContribution']], how='left', left_on='TargetId', right_on='PublicationId').rename(
    columns={'FieldId':'TargetFieldId', 'PubFieldContribution':'TargetPubFieldContribution'})
    del source2target['PublicationId']

    # drop any citation relationships for which we dont have field information
    source2target.dropna(inplace=True)

    # we need to use integer ids to map to the matrix
    source2target[['SourceFieldId', 'TargetFieldId']] = source2target[['SourceFieldId', 'TargetFieldId']].astype(int)

    # in the field2field distance matrix, the weighted contribution from a source publication in multiple fields
    # is the product of the source and target contributions
    source2target['SourcePubFieldContribution'] = source2target['SourcePubFieldContribution'] * source2target['TargetPubFieldContribution']

    # calculate the field representation vectors for this year only
    field2field_mat = dataframe2bipartite(df=source2target, rowname='SourceFieldId', colname='TargetFieldId',
            shape=(Nfields, Nfields), weightname='SourcePubFieldContribution')

    # now normalize the vectors by the source/target counts
    if not count_normalize is None:
        field2field_mat = normalize(field2field_mat, norm='l1', axis=count_normalize)

    return field2field_mat

def field_citation_share(pub2ref, pub2field, count_normalize=None, pub2field_norm=True, temporal=True, 
    citation_direction='references', blocksize = 10**6, show_progress=False):
    """
    Calculate the field citation share based on references or citations.

    Parameters
    ----------
    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2field : DataFrame
        A DataFrame with the field information for each Publication.

    count_normalize : int, default None
        If None: use the raw reference/citation counts.
        If 0: Normalize by the Target Counts
        If 1: Normalize by the Source Counts

    :param pub2field_norm : bool, default True
        When a publication occurs in m > 1 fields, count the publication 1/m times in each field.  Normalizes the membership
        vector so it sums to 1 for each publication.

    :param temporal : bool, default False
        If True, compute the Share matrix using only publications for each year.

    :param citation_direction : str, default `references`
        `references` : the fields are defined by a publication's references.
        `citations` : the fields are defined by a publication's citations.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    CitationShare DataFrame
        if temporal is True
            DataFrame with 4 columns: iFieldId, jFieldId, Year, and Share
        if temporal is False
            DataFrame with 3 columns: iFieldId, jFieldId, Share

    """

    # now we map citing and cited to the source and target depending on which diretion was specified by `citation_direction'
    if citation_direction == 'references':
        pub2ref_rename_dict = {'CitedPublicationId':'TargetId', 'CitingPublicationId':'SourceId'}
        year_col = 'CitingYear'
    elif citation_direction == 'citations':
        pub2ref_rename_dict = {'CitedPublicationId':'SourceId', 'CitingPublicationId':'TargetId'}
        year_col = 'CitedYear'

    required_columns = ['CitedPublicationId', 'CitingPublicationId']
    if temporal:
        required_columns.append(year_col)
    check4columns(pub2ref, required_columns)
    pub2ref = pub2ref[required_columns].dropna().copy(deep=True)

    check4columns(pub2field, ['PublicationId', 'FieldId'])
    pub2field = pub2field.copy(deep=True)

    # to leverage matrix operations we need to map fields to the rows/cols of the matrix
    field2int = {fid:i for i, fid in enumerate(np.sort(pub2field['FieldId'].unique()))}
    int2field = {i:fid for fid, i in field2int.items()}
    pub2field['FieldId'] = [field2int[fid] for fid in pub2field['FieldId'].values]
    Nfields = len(field2int)

    pub2ref.rename(columns=pub2ref_rename_dict, inplace=True)

    # the assignment of a publication to a field is 1/(number of fields) when normalized, and 1 otherwise
    if pub2field_norm:
        pub2nfields = pub2field.groupby('PublicationId')['FieldId'].nunique()
    else:
        pub2nfields = defaultdict(lambda:1)
    pub2field['PubFieldContribution'] = [1.0/pub2nfields[pid] for pid in pub2field['PublicationId'].values]

    citation_share = []

    # differeniate between the temporal and the static RS
    if temporal:

        for y, ydf in pub2ref.groupby(year_col):
            
            yfield2field_mat = field_citation_vectors(ydf, pub2field, n_fields=Nfields, count_normalize=count_normalize)

            nnzrow, nnzcol = np.nonzero(yfield2field_mat)
            for isource, itarget in zip(nnzrow, nnzcol): 
                if isource < itarget:
                    citation_share.append([int2field[isource], int2field[itarget], y, yfield2field_mat[isource, itarget]])

        citation_share = pd.DataFrame(citation_share, columns = ['SourceFieldId', 'TargetFieldId', year_col, 'Share'])

    else:

        field2field_mat = spsparse.coo_matrix( (Nfields, Nfields) )
        
        nref = int(pub2ref.shape[0] / float(blocksize)) + 1
        for itab in range(nref):
            tabdf = pub2ref.loc[itab*int(blocksize):(itab+1)*int(blocksize)]
            
            field2field_mat += field_citation_vectors(tabdf, pub2field, n_fields=Nfields, count_normalize=None)

        # now normalize the vectors by the source/target counts
        if not count_normalize is None:
            field2field_mat = normalize(field2field_mat, norm='l1', axis=count_normalize)

        # now compute the citation_share matrix
        sources, targets = np.nonzero(field2field_mat)
        for isource, itarget in zip(sources, targets):
            citation_share.append([int2field[isource], int2field[itarget], field2field_mat[isource, itarget]])

        citation_share = pd.DataFrame(citation_share, columns = ['SourceFieldId', 'TargetFieldId', 'Share'])

    return citation_share
    
def field_citation_strength(pub2ref, pub2field, count_normalize=None, pub2field_norm=True, temporal=True, 
    citation_direction='references', baseline_year = 1950, show_progress=False):
    """
    Calculate the field citation strength based on references or citations.

    Parameters
    ----------
    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2field : DataFrame
        A DataFrame with the field information for each Publication.

    count_normalize : int, default None
        If None: use the raw reference/citation counts.
        If 0: Normalize by the Target Counts
        If 1: Normalize by the Source Counts

    :param pub2field_norm : bool, default True
        When a publication occurs in m > 1 fields, count the publication 1/m times in each field.  Normalizes the membership
        vector so it sums to 1 for each publication.

    :param temporal : bool, default False
        If True, compute the Strength matrix using only publications for each year.

    :param citation_direction : str, default `references`
        `references` : the fields are defined by a publication's references.
        `citations` : the fields are defined by a publication's citations.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    Strength DataFrame
        if temporal is True
            DataFrame with 4 columns: iFieldId, jFieldId, Year, and Strength
        if temporal is False
            DataFrame with 3 columns: iFieldId, jFieldId, Strength

    """

    if citation_direction == 'references':
        strength_denom = 'TargetFieldId'
        year_col = 'CitingYear'
    elif citation_direction == 'citations':
        strength_denom = 'SourceFieldId'
        year_col = 'CitedYear'

    citation_strength = field_citation_share(pub2ref, pub2field, count_normalize=count_normalize, pub2field_norm=pub2field_norm, 
        temporal=temporal, citation_direction=citation_direction, show_progress=show_progress).rename(columns={'Share':'Strength'})

    pub2field = pub2field[pub2field['Year'] >= baseline_year]

    if temporal:
        for y in np.sort(citation_strength[year_col].unique()):
            yfield_size = pub2field[pub2field['Year'] <= y]['FieldId'].value_counts()

            yfield_size = yfield_size / yfield_size.sum()

            year_rows = citation_strength[year_col] == y
            citation_strength.loc[year_rows, 'Strength'] = [s/yfield_size[normid] for normid, s in citation_strength[year_rows][[strength_denom, 'Strength']].values]

    else:

        field_size = pub2field['FieldId'].value_counts()
        field_size = field_size / field_size.sum()

        citation_strength['Strength'] = [s/field_size[normid] for normid, s in citation_strength[[strength_denom, 'Strength']].values]

    return citation_strength


