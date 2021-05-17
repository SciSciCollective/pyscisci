# -*- coding: utf-8 -*-
"""
.. module:: citationanalysis
    :synopsis: Set of functions for typical bibliometric citation analysis

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

from pyscisci.utils import isin_sorted, groupby_count
from pyscisci.network import cocitation_network

### Novelty

def novelty_conventionality(pubdf, pub2ref_df, focus_pub_ids=None, n_samples = 10, path2randomizednetworks=None, show_progress=False):

    """
    This function calculates the novelty and conventionality for publications as proposed in :cite:`Uzzi2013atypical`.
    

    Parameters
    ----------
    :param pubdf : DataFrame
        A DataFrame with Year and Journal information for each Publication.

    :param pub2ref_df : DataFrame
        A DataFrame with the reference information for each Publication.

    :param focus_pub_ids : list or numpy array, default None
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

    Njournals = len(int2journal)
    years = np.sort(pubdf['Year'].unique())

    temporal_adj = {}
    for y in years:
        yjournal_cite = journalcitation_table.loc[journalcitation_table['CitingYear'] == y]
        yNpubs = yjournal_cite['PublicationId']
        bipartite_adj = dataframe2bipartite(journalcitation_table, 'CitedJournalInt', 'CitingPublicationId', (Njournals, Njournals) )

        adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

        # remove diagonal entries
        adj_mat.setdiag(0)
        adj_mat.eliminate_zeros()

        temporal_adj[y] = adj_mat


    #observed_journal_bipartite = dataframe2bipartite(journalcitation_table, rowname='CitedJournalId', colname='', shape=None, weightname=None)

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