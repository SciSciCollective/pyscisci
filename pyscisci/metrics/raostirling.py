# -*- coding: utf-8 -*-
"""
.. module:: citationanalysis
    :synopsis: Set of functions for typical bibliometric citation analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

import scipy.sparse as spsparse

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from pyscisci.utils import isin_sorted, zip2dict, check4columns
from pyscisci.network import dataframe2bipartite


def field_citation_distance(pub2ref_df, pub2field_df, pub2field_norm=True, temporal=True,citation_direction='references', 
    field_distance_metric='cosine', show_progress=False):
    """
    Calculate the field distance matrix based on references or citations.

    Parameters
    ----------
    :param pub2ref_df : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2field_df : DataFrame
        A DataFrame with the field information for each Publication.

    :param pub2field_norm : bool, default True
        When a publication occurs in m > 1 fields, count the publication 1/m times in each field.  Normalizes the membership
        vector so it sums to 1 for each publication.

    :param temporal : bool, default False
        If True, compute the distance matrix using only publications for each year.

    :param citation_direction : str, default `references`
        `references` : the fields are defined by a publication's references.
        `citations` : the fields are defined by a publication's citations.

    :param field_distance_metric : str, default `cosine`
        The interfield distance metric.  Valid entries come from sklearn.metrics.pairwise_distances:
        ‘cosine‘, ‘euclidean’, ‘l1’, ‘l2’, etc.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    Distance DataFrame
        if temporal is True
            DataFrame with 4 columns: iFieldId, jFieldId, Year, and FieldDistance
        if temporal is False
            DataFrame with 3 columns: iFieldId, jFieldId, FieldDistance

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
    check4columns(pub2ref_df, required_columns)
    pub2ref_df = pub2ref_df[required_columns].dropna().copy(deep=True)

    check4columns(pub2field_df, ['PublicationId', 'FieldId'])
    pub2field_df = pub2field_df.copy(deep=True)

    # to leverage matrix operations we need to map fields to the rows/cols of the matrix
    field2int = {fid:i for i, fid in enumerate(np.sort(pub2field_df['FieldId'].unique()))}
    int2field = {i:fid for fid, i in field2int.items()}
    pub2field_df['FieldId'] = [field2int[fid] for fid in pub2field_df['FieldId'].values]
    Nfields = len(field2int)

    pub2ref_df.rename(columns=pub2ref_rename_dict, inplace=True)

    # the assignment of a publication to a field is 1/(number of fields) when normalized, and 1 otherwise
    if pub2field_norm:
        pub2nfields = pub2field_df.groupby('PublicationId')['FieldId'].nunique()
    else:
        pub2nfields = defaultdict(lambda:1)
    pub2field_df['PubFieldContribution'] = [1.0/pub2nfields[pid] for pid in pub2field_df['PublicationId'].values]

    distance_df = []

    # differeniate between the temporal and the static RS
    if temporal:

        for y, ydf in pub2ref_df.groupby(year_col):
            # merge the references to the fields for the source fields
            ydf = ydf.merge(pub2field_df, how='left', left_on='SourceId', right_on='PublicationId').rename(
            columns={'FieldId':'SourceFieldId', 'PubFieldContribution':'SourcePubFieldContribution'})
            del ydf['PublicationId']

            ydf = ydf.merge(pub2field_df, how='left', left_on='TargetId', right_on='PublicationId').rename(
            columns={'FieldId':'TargetFieldId', 'PubFieldContribution':'TargetPubFieldContribution'})
            del ydf['PublicationId']

            # drop any citation relationships for which we dont have field information
            ydf.dropna(inplace=True)

            # we need to use integer ids to map to the matrix
            ydf[['SourceFieldId', 'TargetFieldId']] = ydf[['SourceFieldId', 'TargetFieldId']].astype(int)

            # in the field2field distance matrix, the weighted contribution from a source publication in multiple fields
            # is the product of the source and target contributions
            ydf['SourcePubFieldContribution'] = ydf['SourcePubFieldContribution'] * ydf['TargetPubFieldContribution']

            # calculate the field representation vectors for this year only
            yfield2field_mat = dataframe2bipartite(df=ydf, rowname='SourceFieldId', colname='TargetFieldId',
                    shape=(Nfields, Nfields), weightname='SourcePubFieldContribution')

            # now compute the distance matrix for this year only
            distance_matrix = pairwise_distances(yfield2field_mat, metric=field_distance_metric)
            nnzrow, nnzcol = np.nonzero(distance_matrix)
            for isource, itarget in zip(nnzrow, nnzcol):
                if isource < itarget:
                    distance_df.append([int2field[isource], int2field[itarget], y, distance_matrix[isource, itarget]])

        distance_df = pd.DataFrame(distance_df, columns = ['iFieldId', 'jFieldId', year_col, 'FieldDistance'])

    else:

        field2field_mat = spsparse.coo_matrix( (Nfields, Nfields) )
        
        nref = int(pub2ref_df.shape[0] / 10.0**6) + 1
        for itab in range(nref):
            tabdf = pub2ref_df.loc[0*10**6:(0+1)*10**6]
            
            tabdf = tabdf.merge(pub2field_df, how='left', left_on='SourceId', right_on='PublicationId').rename(
            columns={'FieldId':'SourceFieldId', 'PubFieldContribution':'SourcePubFieldContribution'})
            del tabdf['PublicationId']

            tabdf = tabdf.merge(pub2field_df, how='left', left_on='TargetId', right_on='PublicationId').rename(
            columns={'FieldId':'TargetFieldId', 'PubFieldContribution':'TargetPubFieldContribution'})
            del tabdf['PublicationId']

            # drop any citation relationships for which we dont have field information
            tabdf.dropna(inplace=True)

            # we need to use integer ids to map to the matrix
            tabdf[['SourceFieldId', 'TargetFieldId']] = tabdf[['SourceFieldId', 'TargetFieldId']].astype(int)

            # in the field2field distance matrix, the weighted contribution from a source publication in multiple fields
            # is the product of the source and target contributions
            tabdf['SourcePubFieldContribution'] = tabdf['SourcePubFieldContribution'] * tabdf['TargetPubFieldContribution']


            # calculate the field representation vectors
            field2field_mat += dataframe2bipartite(df=tabdf, rowname='SourceFieldId', colname='TargetFieldId',
                    shape=(Nfields, Nfields), weightname='SourcePubFieldContribution')

        # now compute the distance matrix
        distance_matrix = pairwise_distances(field2field_mat, metric=field_distance_metric)
        sources, targets = np.nonzero(distance_matrix)
        for isource, itarget in zip(sources, targets):
            if isource < itarget:
                distance_df.append([int2field[isource], int2field[itarget], distance_matrix[isource, itarget]])

        distance_df = pd.DataFrame(distance_df, columns = ['iFieldId', 'jFieldId', 'FieldDistance'])

    return distance_df
    


def raostriling_interdisciplinarity(pub2ref_df, pub2field_df, focus_pub_ids=None, pub2field_norm=True, temporal=False,
    citation_direction='references', field_distance_metric='cosine', distance_matrix=None, show_progress=False):
    """
    Calculate the RaoStirling index as a measure of a publication's interdisciplinarity.
    See :cite:`stirling20` for the definition and :cite:`gates2019naturereach` for an application.

    Parameters
    ----------
    :param pub2ref_df : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2field_df : DataFrame
        A DataFrame with the field information for each Publication.

    :param focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to calculate interdisciplinarity.

    :param pub2field_norm : bool, default True
        When a publication occurs in m > 1 fields, count the publication 1/m times in each field.  Normalizes the membership
        vector so it sums to 1 for each publication.

    :param temporal : bool, default False
        If True, compute the distance matrix using only publications for each year.

    :param citation_direction : str, default `references`
        `references` : the fields are defined by a publication's references.
        `citations` : the fields are defined by a publication's citations.

    :param field_distance_metric : str, default `cosine`
        The interfield distance metric.  Valid entries come from sklearn.metrics.pairwise_distances:
        ‘cosine‘, ‘euclidean’, ‘l1’, ‘l2’, etc.

    :param distance_matrix : numpy array, default None
        The precomputed field distance matrix.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    DataFrame
        DataFrame with 2 columns: 'PublicationId', 'RaoStirling'

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
    check4columns(pub2ref_df, required_columns)
    pub2ref_df = pub2ref_df[required_columns].dropna().copy(deep=True)

    check4columns(pub2field_df, ['PublicationId', 'FieldId'])
    pub2field_df = pub2field_df.copy(deep=True)

    # check that the precomputed distance matrix is the correct size
    if distance_matrix is None:
        distance_matrix = field_citation_distance(pub2ref_df, pub2field_df, pub2field_norm, temporal, 
            citation_direction, field_distance_metric, show_progress)

    field2int = {fid:i for i, fid in enumerate(np.sort(pub2field_df['FieldId'].unique()))}
    pub2field_df['FieldId'] = [field2int[fid] for fid in pub2field_df['FieldId'].values]
    Nfields = len(field2int)

    pub2ref_df.rename(columns=pub2ref_rename_dict, inplace=True)

    if not focus_pub_ids is None:
        pub2ref_df = pub2ref_df.loc[isin_sorted(pub2ref_df['SourceId'].values, focus_pub_ids)]

    if temporal:
        years = np.sort(pub2ref_df[year_col].unique())
        year2int = {y:i for i, y in enumerate(years)}
        Nyears = years.shape[0]

    if type(distance_matrix) == pd.DataFrame and temporal:
        check4columns(distance_matrix, ['iFieldId', 'jFieldId', year_col, 'FieldDistance'])

        distance_matrix = distance_matrix.loc[isin_sorted(distance_matrix[year_col].values, years)].copy(deep=True)

        distance_matrix['iFieldId'] = [field2int.get(fid, None) for fid in distance_matrix['iFieldId'].values]
        distance_matrix['jFieldId'] = [field2int.get(fid, None) for fid in distance_matrix['jFieldId'].values]
        distance_matrix.dropna(inplace=True)

        tdm = np.zeros((Nyears, Nfields, Nfields))
        for y in years:
            tdm[year2int[y]] = dataframe2bipartite(df=distance_matrix[distance_matrix[year_col] == y], rowname='iFieldId', colname='jFieldId',
                shape=(Nfields, Nfields), weightname='FieldDistance').todense()

            tdm[year2int[y]] = tdm[year2int[y]] + tdm[year2int[y]].T

        distance_matrix = tdm


    elif type(distance_matrix) == pd.DataFrame and not temporal:
        check4columns(distance_matrix, ['iFieldId', 'jFieldId', 'FieldDistance'])
        distance_matrix = distance_matrix.copy(deep=True)
        distance_matrix['iFieldId'] = [field2int.get(fid, None) for fid in distance_matrix['iFieldId'].values]
        distance_matrix['jFieldId'] = [field2int.get(fid, None) for fid in distance_matrix['jFieldId'].values]
        distance_matrix.dropna(inplace=True)
        distance_matrix = dataframe2bipartite(df=distance_matrix, rowname='iFieldId', colname='jFieldId',
                shape=(Nfields, Nfields), weightname='FieldDistance').todense()

        distance_matrix = distance_matrix + distance_matrix.T

    elif (type(distance_matrix) == np.array or type(distance_matrix) == np.matrix):
        if not temporal and distance_matrix.shape != (Nfields, Nfields):
            raise pySciSciMetricError('The precomputed_distance_matrix is of the wrong size to compute the RaoStirling interdisciplinarity for the publications passed.')
        elif temporal and distance_matrix.shape != (Nyears, Nfields, Nfields):
            raise pySciSciMetricError('The precomputed_distance_matrix is of the wrong size to compute the RaoStirling interdisciplinarity for the publications and years passed.')

    # the assignment of a publication to a field is 1/(number of fields) when normalized, and 1 otherwise
    if pub2field_norm:
        pub2nfields = pub2field_df.groupby('PublicationId')['FieldId'].nunique()
    else:
        pub2nfields = defaultdict(lambda:1)
    pub2field_df['PubFieldContribution'] = [1.0/pub2nfields[pid] for pid in pub2field_df['PublicationId'].values]

    # merge the references to the fields for the target fields
    pub2ref_df = pub2ref_df.merge(pub2field_df, how='left', left_on='TargetId', right_on='PublicationId').rename(
        columns={'FieldId':'TargetFieldId', 'PubFieldContribution':'TargetPubFieldContribution'})
    del pub2ref_df['PublicationId']

    pub2ref_df.dropna(inplace=True)

    # Now we start on the RaoStiring calculation
    if temporal:

        rsdf = []
        for y, ydf in pub2ref_df.groupby(year_col):
            
            # for each year, we need to map individual publications to the rows of our matrix
            ypub2int = {pid:i for i, pid in enumerate(np.sort(ydf['SourceId'].unique()))}
            yint2pub = {i:pid for pid, i in ypub2int.items()}
            ydf['SourceId'] = [ypub2int[fid] for fid in ydf['SourceId'].values]
            yNpubs = len(ypub2int)

            # calculate the publication representation vectors over fields
            ypub2field_mat = dataframe2bipartite(df=ydf, rowname='SourceId', colname='TargetFieldId',
                shape=(yNpubs, Nfields), weightname='TargetPubFieldContribution').tocsr()

            # make sure the publication 2 field vector is normalized
            ypub2field_mat = normalize(ypub2field_mat, norm='l1', axis=1)

            # finally, we calculate the matrix representation of the RS measure
            yrsdf = pd.DataFrame()
            yrsdf['PublicationId'] = [yint2pub[i] for i in np.sort(ydf['SourceId'].unique())]
            yrsdf['CitingYear'] = y
            yrsdf['RaoStirling'] =  0.5 * np.squeeze(np.asarray(ypub2field_mat.dot(spsparse.csr_matrix(distance_matrix[year2int[y]])).multiply(ypub2field_mat).sum(axis=1)))
            
            rsdf.append(yrsdf)

        rsdf = pd.concat(rsdf)

        return rsdf

    else:

        # first map individual publications to the rows of our matrix
        pub2int = {pid:i for i, pid in enumerate(np.sort(pub2ref_df['SourceId'].unique()))}
        int2pub = {i:pid for pid, i in pub2int.items()}
        pub2ref_df['SourceId'] = [pub2int[pid] for pid in pub2ref_df['SourceId'].values]
        pub2ref_df[['SourceId', 'TargetFieldId']] = pub2ref_df[['SourceId', 'TargetFieldId']].astype(int)
        Npubs = len(pub2int)

        # calculate the publication representation vectors over fields
        pub2field_mat = dataframe2bipartite(df=pub2ref_df, rowname='SourceId', colname='TargetFieldId',
                shape=(Npubs, Nfields), weightname='TargetPubFieldContribution').tocsr()

        # make sure the publication 2 field vector is normalized
        pub2field_mat = normalize(pub2field_mat, norm='l1', axis=1)

        distance_matrix = spsparse.csr_matrix(distance_matrix)

        # finally, we calculate the matrix representation of the RS measure
        rsdf = pd.DataFrame()
        rsdf['RaoStirling'] = 0.5 * np.squeeze(np.asarray( spsparse.csr_matrix.multiply(pub2field_mat.dot(distance_matrix), pub2field_mat).sum(axis=1)))
        rsdf['PublicationId'] = [int2pub[i] for i in np.sort(pub2ref_df['SourceId'].unique())]
        
        return rsdf

