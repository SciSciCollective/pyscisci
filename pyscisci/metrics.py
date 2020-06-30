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
import pandas as pd
import numpy as np

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.utils import isin_sorted, zip2dict, check4columns, fit_piecewise_linear, groupby_count, groupby_range, rank_array
from pyscisci.network import cocitation_network

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

###
def RaoStriling(data_array, distance_matrix):
    rs = 0.0
    normed_data = data_array / data_array.sum()
    for ip, jp in itertools.combinations(np.nonzero(data_array)[0], 2):
        rs += distance_matrix[ip, jp] * normed_data[ip] * normed_data[jp]

    return rs

def Interdisciplinarity(pub2ref, pub2field, pub2year=None, temporal=True, citation_direction='ref'):
    """
    Calculate the h index for each group in the DataFrame.  See :cite:`hirsch2005index` for the definition
    and :cite:`gates2019naturereach` for an application.

    Parameters
    ----------
    :param pub2ref : DataFrame
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

    raise NotImplementedError

    field2int = {fid:i for i, fid in enumerate(np.sort(pub2field.unique()))}

    Nfields = len(field2int)

    pass

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

    raise NotImplementedError

    journalcitation_table, int2journal = create_journalcitation_table(pubdf, pub2ref)

    for isample in range(n_samples):
        database_table = database_table.groupby(['CitingYear', 'CitedYear'], sort=False)['CitedJournalInt'].transform(np.random.permutation)

