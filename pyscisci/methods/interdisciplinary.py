# -*- coding: utf-8 -*-
"""
.. module:: interdisciplinary
    :synopsis: Set of functions for typical interdisciplinary analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

from ..utils import isin_sorted, zip2dict, check4columns, simpson, simpson_finite, shannon_entropy



def simpson_interdisciplinarity(pub2ref, pub2field, focus_pub_ids=None,
    citation_direction='references', finite_correction=False, show_progress=False):
    """
    Calculate the Simpson index as a measure of a publication's interdisciplinarity.
    See :cite:`stirling20` for the definition.

    Parameters
    ----------
    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2field : DataFrame
        A DataFrame with the field information for each Publication.

    :param focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to calculate interdisciplinarity.

    :param finite_correction : bool, default False
        Whether to apply the correction for a finite sample.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    DataFrame
        DataFrame with 2 columns: 'PublicationId', 'Simpsons'

    """

    # now we map citing and cited to the source and target depending on which diretion was specified by `citation_direction'
    if citation_direction == 'references':
        pub2ref_rename_dict = {'CitedPublicationId':'TargetId', 'CitingPublicationId':'SourceId'}
        year_col = 'CitingYear'
    elif citation_direction == 'citations':
        pub2ref_rename_dict = {'CitedPublicationId':'SourceId', 'CitingPublicationId':'TargetId'}
        year_col = 'CitedYear'

    required_columns = ['CitedPublicationId', 'CitingPublicationId']
    check4columns(pub2ref, required_columns)
    pub2ref = pub2ref[required_columns].rename(columns=pub2ref_rename_dict)

    check4columns(pub2field, ['PublicationId', 'FieldId'])

    # merge the references to the fields for the target fields
    pub2ref = pub2ref.merge(pub2field, how='left', left_on='TargetId', 
        right_on='PublicationId').rename(columns={'FieldId':'TargetFieldId'})
    del pub2ref['PublicationId']

    pub2ref = pub2ref.dropna()
    
    if finite_correction:
        simpdf = 1-pub2ref.groupby('SourceId')['TargetFieldId'].apply(simpson_finite)
    else:
        simpdf = 1-pub2ref.groupby('SourceId')['TargetFieldId'].apply(simpson)

    simpdf = simpdf.to_frame().reset_index().rename(
        columns={'TargetFieldId':'SimpsonInterdisciplinarity', 'SourceId':'PublicationId'})
    
    return simpdf


def shannon_interdisciplinarity(pub2ref, pub2field, focus_pub_ids=None,
    citation_direction='references', normalized=False, K=None, show_progress=False):
    """
    Calculate the Shannon entropy as a measure of a publication's interdisciplinarity.
    See :cite:`stirling20` for the definition.

    Parameters
    ----------
    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2field : DataFrame
        A DataFrame with the field information for each Publication.

    :param focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to calculate interdisciplinarity.

    :param temporal : bool, default False
        If True, compute the distance matrix using only publications for each year.

    :param normalized : bool, default False
        If True, use the normalized entorpy bounded by the number of observed fields
        or K if not None.
    
    :param K : int, default None
        The maximum number of fields to consider.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    DataFrame
        DataFrame with 2 columns: 'PublicationId', 'Simpsons'

    """

    # now we map citing and cited to the source and target depending on which diretion was specified by `citation_direction'
    if citation_direction == 'references':
        pub2ref_rename_dict = {'CitedPublicationId':'TargetId', 'CitingPublicationId':'SourceId'}
        year_col = 'CitingYear'
    elif citation_direction == 'citations':
        pub2ref_rename_dict = {'CitedPublicationId':'SourceId', 'CitingPublicationId':'TargetId'}
        year_col = 'CitedYear'

    required_columns = ['CitedPublicationId', 'CitingPublicationId']
    check4columns(pub2ref, required_columns)
    pub2ref = pub2ref[required_columns].rename(columns=pub2ref_rename_dict)

    check4columns(pub2field, ['PublicationId', 'FieldId'])

    if K is None:
        K = pub2field['FieldId'].nunique()

    # merge the references to the fields for the target fields
    pub2ref = pub2ref.merge(pub2field, how='left', left_on='TargetId', 
        right_on='PublicationId').rename(columns={'FieldId':'TargetFieldId'})
    del pub2ref['PublicationId']

    pub2ref = pub2ref.dropna()
    
    shan_inter = pub2ref.groupby('SourceId')['TargetFieldId'].apply(shannon_entropy)
    shan_inter = shan_inter.to_frame().reset_index().rename(
        columns={'TargetFieldId':'ShannonInterdisciplinarity', 'SourceId':'PublicationId'})

    if normalized:
        shan_inter['ShannonInterdisciplinarity'] = shan_inter['ShannonInterdisciplinarity']/np.log(K)
    
    return shan_inter

