# -*- coding: utf-8 -*-
"""
.. module:: novelty
    :synopsis: Set of functions to quantify the novelty and conventionality

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np
from itertools import combinations

from pyscisci.utils import isin_sorted, groupby_count, value_to_int, welford_mean_m2
from pyscisci.network import cocitation_network

### Novelty

def novelty_conventionality(pub, pub2ref, focus_pub_ids=None, n_samples = 10, show_progress=False):

    """
    This function calculates the novelty and conventionality for publications as proposed in :cite:`Uzzi2013atypical`.
    
    ToDo: path2randomnetwork, save randomizations


    Parameters
    ----------
    pub : DataFrame
        A DataFrame with Year and Journal information for each Publication.

    pub2ref : DataFrame
        A DataFrame with the reference information for each Publication.

    focus_pub_ids : list or numpy array, default None
        A list of PublicationIds for which to compute the novelty score.

    n_samples : int, default 10
        The number of randomized networks in the ensemble.

    show_progress : bool, default False
        If True, show a progress bar tracking the calculation.

    Returns
    -------
    DataFrame
        The original dataframe with a new column for rank: colrankby+"Rank"

    """

    raise NotImplementedError

    journal2citation_table, int2journal = create_journalcitation_table(pub, pub2ref)
    Njournals = len(int2journal)

    if isinstance(focus_pub_ids, list) or isinstance(focus_pub_ids, np.array):
        focus_pub_ids = np.sort(focus_pub_ids)
    elif isinstance(focus_pub_ids, int) or isinstance(focus_pub_ids, str):
        focus_pub_ids = np.sort([focus_pub_ids])

    if focus_pub_ids is None:
        years = pub['Year'].unique()
    else:
        years = np.sort(pub.loc[isin_sorted(pub['PublicationId'].values, focus_pub_ids)]['Year'].unique())


    focus_ref_distribution = []
    for y in years:
        yjournal_cite = journalcitation_table.loc[journalcitation_table['CitingYear'] == y]
        
        observed_journal_bipartite = dataframe2bipartite(journalcitation_table, 'CitedJournalInt', 'CitingPublicationId', (Njournals, Njournals) )
        observed_journal_adj = project_bipartite_mat(observed_journal_bipartite, project_to = 'row').todense()

        database_table = yjournal_cite.copy()

        random_means = np.zeros((Njournals, Njournals))
        random_m2s = np.zeros((Njournals, Njournals))

        for isample in range(n_samples):
            database_table['CitedJournalInt'] = database_table.groupby(['CitingYear', 'CitedYear'], sort=False)['CitedJournalInt'].transform(np.random.permutation)
            random_journal_bipartite = dataframe2bipartite(database_table, rowname='CitedJournalId', colname='CitingPublicationId', shape=(Njournals, Njournals))
            random_journal_adj = project_bipartite_mat(random_journal_bipartite, project_to = 'row').todense()

            # update our running means and m2 (mean squared errors) using Welford's algorithm
            _, random_means, random_m2s = welford_mean_m2(isample, random_means, random_m2s, random_journal_adj)

        journal_z_scores = (observed_journal_adj - random_means)/np.sqrt(random_m2s/n_samples)

        yfocus_ref = yjournal_cite.loc[isin_sorted(pub2ref['CitingPublicationId'].values, focus_pub_ids)]
        def pair_of_ref(reflist):
            itertools.combinations(reflist, 2)
        #focus_ref_distribution.extend([[fid, rid1, rid2, journal_z_scores[rj1, rj2]] for fid, rid1, rid2, rj1, rj2 in yfocus_ref[['CitingPublicationId']]])

        #yfocus_ref.groupby('CitingPublicationId')[''].apply()


    

def create_journalcitation_table(pub, pub2ref):
    required_pub_columns = ['PublicationId', 'JournalId', 'Year']
    check4columns(pub, required_pub_columns)
    pub = pub[required_pub_columns]

    required_pub2ref_columns = ['CitingPublicationId', 'CitedPublicationId']
    check4columns(pub2ref, required_pub_columns)
    pub2ref = pub2ref[required_pub2ref_columns]

    pub['JournalInt'], journal2int = value_to_int(pub['JournalId'].values, sort_values='value', return_map=True)

    jctable = pub2ref.merge(pub[['PublicationId', 'Year', 'JournalInt']], how='left', left_on = 'CitingPublicationId', right_on = 'PublicationId')
    jctable.rename({'Year':'CitingYear', 'JournalInt':'CitingJournalInt'})
    del jctable['PublicationId']
    #del jctable['CitingPublicationId']

    jctable = jctable.merge(pub[['PublicationId', 'Year', 'JournalInt']], how='left', left_on = 'CitedPublicationId', right_on = 'PublicationId')
    jctable.rename({'Year':'CitedYear', 'JournalInt':'CitedJournalInt'})
    del jctable['PublicationId']
    #del jctable['CitedPublicationId']


    return jctable, {i:j for j,i in journal2int.items()}