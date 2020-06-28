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

