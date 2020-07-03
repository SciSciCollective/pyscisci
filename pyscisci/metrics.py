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

def compute_citation_rank(df, colgroupby='Year', colrankby='C10', ascending=True, normed=False, show_progress=False):
    """
    Rank elements in the array from 0 (smallest) to N -1 (largest)

    Parameters
    ----------
    :param df : DataFrame
        A DataFrame with the citation information for each Publication.

    :param colgroupby : str, list
        The DataFrame column(s) to subset by.

    :param colrankby : str
        The DataFrame column to rank by.

    :param ascending : bool, default True
        Sort ascending vs. descending.

    :param normed : bool, default False
        False : rank is from 0 to N -1
        True : rank is from 0 to 1

    :param show_progress : bool, default False
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

def author_hindex(a):
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
    return df.groupby(colgroupby, sort=False)[colcountby].progress_apply(author_hindex).to_frame().reset_index().rename(columns=newname_dict)


def pub_credit_share(focus_pid, pub2ref_df, pub2author_df, temporal=False, normed=False, show_progress=False):
    """
    Calculate the credit share for each author of a publication.

    References
    ----------
    .. [w] Shen, Barabasi (2014): "Collective credit allocation in science", *PNAS*. 111, 12325-12330.
           DOI: 10.1073/pnas.1401992111

    Parameters
    ----------
    :param focus_pid : int, str
        The focus publication id.

    :param pub2ref_df : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2author_df : DataFrame
        A DataFrame with the author information for each Publication.

    :param temporal : bool, default False
        If True, compute the adjacency matrix using only publications for each year.

    :param normed : bool, default False
        Normalize the sum of credit share to 1.0

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    credit_share, numpy array
        If temporal == False:
            The adjacency matrix for the co-citation network

        If temporal == True:
            A dictionary with key for each year, and value of the adjacency matrix for the cocitation network induced
            by citing publications in that year.

    author2int, dict
        A mapping of the AuthorIds from the focus publication to the column of the credit share vector or matrix (see above).

    """

    # start by getting the co-citation network around the focus publication
    adj_mat, cited2int = cocitation_network(pub2ref_df, focus_pub_ids=np.sort([focus_pid]), focus_constraint='egocited',
            temporal=temporal, show_progress=show_progress)

    # get the authorships for the publications in the cocitation network
    cocited_pubs = np.sort(list(cited2int.keys()))
    pa_df = pub2author_df.loc[isin_sorted(pub2author_df['PublicationId'].values, cocited_pubs)]

    # the focus publication's authors
    focus_authors = np.sort(pa_df.loc[pa_df['PublicationId']==focus_pid]['AuthorId'].unique())
    author2int = {aid:i for i, aid in enumerate(focus_authors)}

    # the credit allocation matrix has a row for each focus author, and a column for each cocited publication (including the focus pub)
    credit_allocation_mat = np.zeros((focus_authors.shape[0], cocited_pubs.shape[0]), dtype = float)

    # for each cocited publication, we count the number of authors
    # and assign to each focus author, their fractional share of the credit (1 divided by the number of authors)
    for cocitedid, adf in pa_df.groupby('PublicationId'):
        author2row = [author2int[aid] for aid in adf['AuthorId'].unique() if not author2int.get(aid, None) is None]
        if len(author2row) > 0:
            credit_allocation_mat[author2row, cited2int[cocitedid]] = 1.0/adf['AuthorId'].nunique()

    if temporal:
        # temporal credit allocation - broken down by year

        # we need the temporal citations to the focus article
        focus_citations = groupby_count(pub2ref_df.loc[isin_sorted(pub2ref_df['CitedPublicationId'].values, np.sort([focus_pid]))],
            colgroupby='CitingYear', colcountby='CitingPublicationId', count_unique=True, show_progress=False)
        focus_citations={y:c for y,c in focus_citations[['CitingYear', 'CitingPublicationIdCount']].values}

        # when temporal is True, a temporal adj mat is returned where each key is the year
        years = np.sort(list(adj_mat.keys()))

        cocite_counts = np.zeros((years.shape[0], cocited_pubs.shape[0]), dtype=float)

        for iy, y in enumerate(years):
            cocite_counts[iy] = adj_mat[y].tocsr()[cited2int[focus_pid]].todense()
            cocite_counts[iy, cited2int[focus_pid]] = focus_citations[y]

        cocite_counts = cocite_counts.cumsum(axis=0)

    else:
        # just do credit allocation with the full cocitation matrix
        cocite_counts = adj_mat.tocsr()[cited2int[focus_pid]].todense()

        # the co-citation matrix misses the number of citations to the focus publication
        # so explicitly calculate the number of citations to the focus publication
        cocite_counts[0,cited2int[focus_pid]] = pub2ref_df.loc[isin_sorted(pub2ref_df['CitedPublicationId'].values, np.sort([focus_pid]))]['CitingPublicationId'].nunique()

    # credit share is the matrix product of the credit_allocation_mat with cocite_counts
    credit_share = np.squeeze(np.asarray(credit_allocation_mat.dot(cocite_counts.T)))

    # normalize the credit share vector to sum to 1
    if normed:
        credit_share = credit_share/credit_share.sum(axis=0)

    if temporal:
        return credit_share, author2int, years
    else:
        return credit_share, author2int

### Productivity Trajectory

def _fit_piecewise_lineardf(author_df, args):
    return fit_piecewise_linear(author_df[args[0]].values, author_df[args[1]].values)

def compute_yearly_productivity_traj(df, colgroupby = 'AuthorId', colx='Year',coly='YearlyProductivity'):
    """
    This function calculates the piecewise linear yearly productivity trajectory original studied in [w].

    References
    ----------
    .. [w] Way, Larremore (2018): "title", *PNAS*.
           DOI: xxx
    """

    newname_dict = zip2dict(list(range(4)), ['t_break', 'b', 'm1', 'm2' ]) #[str(i) for i in range(4)]
    return df.groupby(colgroupby, sort=False).apply(_fit_piecewise_lineardf, args=(colx,coly) ).reset_index().rename(columns = newname_dict)

### Disruption
def compute_disruption_index(pub2ref, show_progress=False):
    """
    Funk, Owen-Smith (2017) A Dynamic Network Measure of Technological Change *Management Science* **63**(3),791-817
    Wu, Wang, Evans (2019) Large teams develop and small teams disrupt science and technology *Nature* **566**, 378–382

    """
    if show_progress:
        print("Starting computation of disruption index.")

    reference_groups = pub2ref.groupby('CitingPublicationId', sort = False)['CitedPublicationId']
    citation_groups = pub2ref.groupby('CitedPublicationId', sort = False)['CitingPublicationId']

    def get_citation_groups(pid):
        try:
            return citation_groups.get_group(pid).values
        except KeyError:
            return []

    def disruption_index(citing_focus):
        focusid = citing_focus.name

        # if the focus publication has no references, then it has a disruption of None
        try:
            focusref = reference_groups.get_group(focusid)
        except KeyError:
            return None

        # implementation 1: keep it numpy
        #cite2ref = reduce(np.union1d, [get_citation_groups(refid) for refid in focusref])
        #nj = np.intersect1d(cite2ref, citing_focus.values).shape[0]
        #nk = cite2ref.shape[0] - nj

        # implementation 2: but dicts are faster...
        cite2ref = {citeid:1 for refid in focusref for citeid in get_citation_groups(refid)}
        nj = sum(cite2ref.get(pid, 0) for pid in citing_focus.values )
        nk = len(cite2ref) - nj

        ni = citing_focus.shape[0] - nj

        return (ni - nj)/(ni + nj + nk)

    # register our pandas apply with tqdm for a progress bar
    tqdm.pandas(desc='Disruption Index', disable= not show_progress)

    newname_dict = {'CitingPublicationId':'DisruptionIndex', 'CitedPublicationId':'PublicationId'}
    return citation_groups.progress_apply(disruption_index).to_frame().reset_index().rename(columns = newname_dict)


### Cnorm
def compute_cnorm(pub2ref, pub2year):
    """
    This function calculates the cnorm for publications.

    References
    ----------
    .. [h] Ke, Q., Gates, A. J., Barabasi, A.-L. (2020): "title",
           *in submission*.
           DOI: xxx
    """
    raise NotImplementedError

    required_pub2ref_columns = ['CitingPublicationId', 'CitedPublicationId']
    check4columns(pub2ref, required_pub_columns)
    pub2ref = pub2ref[required_pub2ref_columns]

    # we need the citation counts and cocitation network
    temporal_cocitation_dict = {y:defaultdict(set) for y in set(pub2year.values())}
    temporal_citation_dict = {y:defaultdict(int) for y in temporal_cocitation_dict.keys()}

    def count_cocite(cited_df):
        y = pub2year[cited_df.name]

        for citedpid in cited_df['CitedPublicationId'].values:
            temporal_citation_dict[y][citedpid] += 1
        for icitedpid, jcitedpid in combinations(cited_df['CitedPublicationId'].values, 2):
            temporal_cocitation_dict[y][icitedpid].add(jcitedpid)
            temporal_cocitation_dict[y][jcitedpid].add(icitedpid)

    pub2ref.groupby('CitingPublicationId', sort=False).apply(count_cocite)

    cnorm = {}
    for y in temporal_citation_dict.keys():
        for citedpid, year_cites in temporal_citation_dict[y].items():
            if cnorm.get(citedpid, None) is None:
                cnorm[citedpid] = {y:year_cites/np.mean()}

def compute_raostriling_interdisciplinarity(pub2ref_df, pub2field_df, focus_pub_ids=None, pub2field_norm=True, temporal=False,
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

    required_columns = ['CitedPublicationId', 'CitingPublicationId']
    if temporal:
        required_columns.append('CitingYear')
    check4columns(pub2ref_df, required_columns)
    pub2ref_df = pub2ref_df[required_columns].dropna()

    check4columns(pub2field_df, ['PublicationId', 'FieldId'])

    # to leverage matrix operations we need to map fields to the rows/cols of the matrix
    field2int = {fid:i for i, fid in enumerate(np.sort(pub2field_df['FieldId'].unique()))}
    pub2field_df['FieldId'] = [field2int[fid] for fid in pub2field_df['FieldId'].values]
    Nfields = len(field2int)

    if temporal:
        years = np.sort(pub2ref_df['CitingYear'].unique())
        year2int = {y:i for i, y in enumerate(years)}
        Nyears = years.shape[0]

    # check that the precomputed distance matrix is the correct size
    if not precomputed_distance_matrix is None:
        if not temporal and precomputed_distance_matrix != (Nfields, Nfields):
            raise pySciSciMetricError('The precomputed_distance_matrix is of the wrong size to compute the RaoStirling interdisciplinarity for the publications passed.')
        elif temporal and precomputed_distance_matrix != (Nyears, Nfields, Nfields):
            raise pySciSciMetricError('The precomputed_distance_matrix is of the wrong size to compute the RaoStirling interdisciplinarity for the publications and years passed.')

    # the assignment of a publication to a field is 1/(number of fields) when normalized, and 1 otherwise
    if pub2field_norm:
        pub2nfields = pub2field_df.groupby('PublicationId')['FieldId'].nunique()
    else:
        pub2nfields = defaultdict(lambda:1)
    pub2field_df['PubFieldContribution'] = [1.0/pub2nfields[pid] for pid in pub2field_df['PublicationId'].values]

    # now we map citing and cited to the source and target depending on which diretion was specified by `citation_direction'
    if citation_direction == 'references':
        pub2ref_rename_dict = {'CitedPublicationId':'TargetId', 'CitingPublicationId':'SourceId'}
    elif citation_direction == 'citations':
        pub2ref_rename_dict = {'CitedPublicationId':'SourceId', 'CitingPublicationId':'TargetId'}

    pub2ref_df = pub2ref_df.rename(columns=pub2ref_rename_dict)

    # merge the references to the fields for the target fields
    pub2ref_df = pub2ref_df.merge(pub2field_df, how='left', left_on='TargetId', right_on='PublicationId').rename(
        columns={'FieldId':'TargetFieldId', 'PubFieldContribution':'TargetPubFieldContribution'})
    del pub2ref_df['PublicationId']

    # we need to calcuate the field 2 field distance matrix
    if distance_matrix is None:

        # merge the references to the fields for the source fields
        pub2ref_df = pub2ref_df.merge(pub2field_df, how='left', left_on='SourceId', right_on='PublicationId').rename(
        columns={'FieldId':'SourceFieldId', 'PubFieldContribution':'SourcePubFieldContribution'})
        del pub2ref_df['PublicationId']

        # drop any citation relationships for which we dont have field information
        pub2ref_df.dropna(inplace=True)

        # we need to use integer ids to map to the matrix
        pub2ref_df[['SourceFieldId', 'TargetFieldId']] = pub2ref_df[['SourceFieldId', 'TargetFieldId']].astype(int)

        # in the field2field distance matrix, the weighted contribution from a source publication in multiple fields
        # is the product of the source and target contributions
        pub2ref_df['SourcePubFieldContribution'] = pub2ref_df['SourcePubFieldContribution'] * pub2ref_df['TargetPubFieldContribution']

        # differeniate between the temporal and the static RS
        if temporal:
            # make the temporal distance matrix
            distance_matrix = np.zeros((Nyears, Nfields, Nfields))

            for y, ydf in pub2ref_df.groupby('CitingYear'):
                # calculate the field representation vectors for this year only
                yfield2field_mat = dataframe2bipartite(df=ydf, rowname='SourceFieldId', colname='TargetFieldId',
                        shape=(Nfields, Nfields), weightname='SourcePubFieldContribution')

                # now compute the distance matrix for this year only
                distance_matrix[year2int[y]] = pairwise_distances(yfield2field_mat, metric=field_distance_metric)

        else:
            # calculate the field representation vectors
            field2field_mat = dataframe2bipartite(df=pub2ref_df, rowname='SourceFieldId', colname='TargetFieldId',
                shape=(Nfields, Nfields), weightname='SourcePubFieldContribution')

            # now compute the distance matrix
            distance_matrix = pairwise_distances(field2field_mat, metric=field_distance_metric)

        # we no longer need the 'SourceFieldId' or 'SourcePubFieldContribution' so cleanup
        del pub2ref_df['SourceFieldId']
        del pub2ref_df['SourcePubFieldContribution']
        pub2ref_df.drop_duplicates(subset=['SourceId', 'TargetId', 'TargetFieldId'], inplace=True)


    # Now we start on the RaoStiring calculation

    # drop any citation relationships for which we dont have field information
    pub2ref_df.dropna(inplace=True)

    if temporal:

        rsdf = []
        for y, ydf in pub2ref_df.groupby('CitingYear'):

            # for each year, we need to map individual publications to the rows of our matrix
            ypub2int = {pid:i for i, pid in enumerate(np.sort(ydf['SourceId'].unique()))}
            ydf['SourceId'] = [ypub2int[fid] for fid in ydf['SourceId'].values]
            ydf[['SourceId', 'TargetFieldId']] = ydf[['SourceId', 'TargetFieldId']].astype(int)
            yNpubs = len(ypub2int)

            # calculate the publication representation vectors over fields
            ypub2field_mat = dataframe2bipartite(df=ydf, rowname='SourceId', colname='TargetFieldId',
                shape=(yNpubs, Nfields), weightname='TargetPubFieldContribution').tocsr()

            # make sure the publication 2 field vector is normalized
            ypub2field_mat = normalize(ypub2field_mat, norm='l1', axis=1)

            # finally, we calculate the matrix representation of the RS measure
            yrsdf = 0.5 * np.squeeze(np.asarray(ypub2field_mat.dot(distance_matrix[year2int[y]]).multiply(ypub2field_mat).sum(axis=1)))

            rsdf.append(pd.DataFrame(zip(np.sort(ydf['SourceId'].unique()), yrsdf, [y]*yNpubs), columns = ['PublicationId', 'RaoStirling', 'CitingYear']))

        rsdf = pd.concat(rsdf)

        return rsdf, precomputed_distance_matrix, field2int, years

    else:

        # first map individual publications to the rows of our matrix
        pub2int = {pid:i for i, pid in enumerate(np.sort(pub2ref_df['SourceId'].unique()))}
        pub2ref_df['SourceId'] = [pub2int[fid] for fid in pub2ref_df['SourceId'].values]
        pub2ref_df[['SourceId', 'TargetFieldId']] = pub2ref_df[['SourceId', 'TargetFieldId']].astype(int)
        Npubs = len(pub2int)

        # calculate the publication representation vectors over fields
        pub2field_mat = dataframe2bipartite(df=pub2ref_df, rowname='SourceId', colname='TargetFieldId',
                shape=(Npubs, Nfields), weightname='TargetPubFieldContribution').tocsr()

        # make sure the publication 2 field vector is normalized
        pub2field_mat = normalize(pub2field_mat, norm='l1', axis=1)

        # finally, we calculate the matrix representation of the RS measure
        rsdf = 0.5 * np.squeeze(np.asarray(pub2field_mat.dot(distance_matrix).multiply(pub2field_mat).sum(axis=1)))

        rsdf = pd.DataFrame(zip(np.sort(pub2ref_df['SourceId'].unique()), rsdf), columns = ['PublicationId', 'RaoStirling'])

        return rsdf, distance_matrix, field2int



### Novelty

def compute_novelty(pubdf, pub2ref_df, focuspubids=None, n_samples = 10, path2randomizednetworks=None, show_progress=False):
    
    """
    This function calculates the novelty and conventionality for publications.
    References
    ----------
    .. [u] Uzzi, B., Mukherjee, S., Stringer, M. and Jones, B. (2013): "Atypical Combinations and Scientific Impact",
           *Science*. Vol. 342, Issue 6157, pp. 468-472
           DOI: 10.1126/science.1240474

    Parameters
    ----------
    :param pubdf : DataFrame
        A DataFrame with Year and Journal information for each Publication.

    :param pub2ref_df : DataFrame
        A DataFrame with the reference information for each Publication.

    :param focuspubids : list or numpy array, default None
        A list of PublicationIds for which to compute the novelty score.

    :param n_samples : int, default 10
        The number of randomized networks in the ensemble.

    :param path2randomizednetworks : str, default None
        The Novelty calculation requires an ensemble of randomized networks.  If a path is specified by path2randomizednetworks, this
        will first check if any randomized networks exists.  Alternatively, if the directory specified by path2randomizednetworks is empty,
        then any randomized networks will be saved here.

    :param normed : bool, default False
        False : rank is from 0 to N -1
        True : rank is from 0 to 1

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    DataFrame
        The original dataframe with a new column for rank: colrankby+"Rank"

    """

    raise NotImplementedError

    journalcitation_table, int2journal = create_journalcitation_table(pubdf, pub2ref)

    years = np.sort(paa_df['Year'].unique())

    temporal_adj = {}
    for y in years:
        bipartite_adj = dataframe2bipartite(journalcitation_table.loc[journalcitation_table['CitingYear'] == y], 'AuthorId', 'PublicationId', (Nauthors, Npubs) )
        
        adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

        # remove diagonal entries
        adj_mat.setdiag(0)
        adj_mat.eliminate_zeros()

        temporal_adj[y] = adj_mat 


    observed_journal_bipartite = dataframe2bipartite(journalcitation_table, rowname='CitedJournalId', colname='', shape=None, weightname=None)

    for isample in range(n_samples):
        database_table = database_table.groupby(['CitingYear', 'CitedYear'], sort=False)['CitedJournalInt'].transform(np.random.permutation)

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


class pySciSciMetricError(Exception):
    """
    Base Class for metric errors.
    """
    def __str__(self, msg=None):
        if msg is None:
            return 'pySciSci metric error.'
        else:
            return msg
