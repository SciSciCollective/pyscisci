# -*- coding: utf-8 -*-
"""
.. module:: novelty
    :synopsis: Set of functions to quantify the novelty and conventionality

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

from itertools import combinations

from pyscisci.utils import isin_sorted, groupby_count, value_to_int, welford_mean_m2, zscore_var, check4columns
from pyscisci.sparsenetworkutils import dataframe2bipartite, project_bipartite_mat

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
import sys
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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

    desc=''
    if isinstance(show_progress, str):
        desc = show_progress

    #first subset only to the data columns we need
    required_pub_columns = ['PublicationId', 'Year', 'JournalId']
    check4columns(pub, required_pub_columns)
    pub = pub[required_pub_columns].dropna()

    required_pub2ref_columns = ['CitingPublicationId', 'CitedPublicationId']
    check4columns(pub2ref, required_pub2ref_columns)
    pub2ref = pub2ref[required_pub2ref_columns]

    # now map the journal values to integers so we can use them for matrix indices
    pub['JournalInt'], journal2int = value_to_int(pub['JournalId'].values, sort_values='value', return_map=True)
    Njournals = len(journal2int)

    # now map the publication values to integers so we can use them for matrix indices
    pub['PublicationInt'], pub2int = value_to_int(pub['PublicationId'].values, sort_values='value', return_map=True)
    Npubs = pub['PublicationInt'].nunique()
    int2pub = {i:pid for pid, i in pub2int.items()}

    pub2ref = pub2ref.merge(pub[['PublicationId', 'PublicationInt', 'Year']], how='left', left_on='CitingPublicationId', right_on='PublicationId')
    del pub2ref['PublicationId']
    pub2ref.rename(columns={'PublicationInt':'CitingPublicationInt', 'Year':'CitingYear'}, inplace=True)

    pub2ref = pub2ref.merge(pub[['PublicationId', 'Year', 'JournalInt']], how='left', left_on='CitedPublicationId', right_on='PublicationId')
    del pub2ref['PublicationId']
    del pub2ref['CitedPublicationId']
    pub2ref.rename(columns={'JournalInt':'CitedJournalInt', 'Year':'CitedYear'}, inplace=True)

    pub2ref = pub2ref.dropna().reset_index(drop=True)

    # book keeping for our focus publications
    if isinstance(focus_pub_ids, list) or isinstance(focus_pub_ids, np.ndarray):
        focus_pub_ids = np.sort(focus_pub_ids)
    elif isinstance(focus_pub_ids, int) or isinstance(focus_pub_ids, str):
        focus_pub_ids = np.sort([focus_pub_ids])

    # book keeping for the focus years
    if focus_pub_ids is None:
        years = np.sort(pub2ref['CitingYear'].unique())
    else:
        years = np.sort(pub2ref[isin_sorted(pub2ref['CitingPublicationId'].values, focus_pub_ids)]['CitingYear'].unique())

    # The scores are found for all publications in one year, so we can treat each year independently
    novelty_score = []
    for citing_year in tqdm(years, desc="Novelty_Conventoinality", leave=True, disable=not show_progress):

        # subset to the focus year
        ypub2ref = pub2ref.loc[pub2ref['CitingYear'] == citing_year].reset_index(drop=True)
        
        # find the observed journal-pair frequencies
        observed_journal_bipartite = dataframe2bipartite(ypub2ref, 'CitedJournalInt', 'CitingPublicationInt', (Njournals, Npubs) )
        observed_journal_adj = project_bipartite_mat(observed_journal_bipartite, project_to = 'row').todense()

        # copy over the table
        database_table = ypub2ref.copy()

        random_means = np.zeros((Njournals, Njournals))
        random_m2s = np.zeros((Njournals, Njournals))

        # run this loop for each sample
        for isample in range(n_samples):

            # the MCMC step discussed in the Supplemental for :cite:`Uzzi2013atypical` is equivalent to a hard-shuffle of the journal column 
            database_table['CitedJournalInt'] = database_table.groupby(['CitedYear'], sort=False)['CitedJournalInt'].transform(np.random.permutation)
            
            # now find the random journal pair frequencies
            random_journal_bipartite = dataframe2bipartite(database_table, rowname='CitedJournalInt', colname='CitingPublicationInt', shape=(Njournals, Npubs))
            random_journal_adj = project_bipartite_mat(random_journal_bipartite, project_to = 'row').todense()

            # update our running means and m2 (mean squared errors) using Welford's algorithm
            # this means we dont have to keep the random networks in memory and instead can just update our statistics for each sample!
            _, random_means, random_m2s = welford_mean_m2(isample, random_means, random_m2s, random_journal_adj)
        
        # find the journal pair z-scores   
        # a lot of the entries will still be zero, so lets hide the warnings when we calculate the z-score
        with np.errstate(divide='ignore',invalid='ignore'):
            journal_zscores = zscore_var(observed_journal_adj, random_means, random_m2s/n_samples)

        # now we need to map over the references to each publication
        def _ref_distribution(refentry):
            if refentry.shape[0] > 1:
                # find the zscore for each pair of references
                ref_pair_zscores = [journal_zscores[i,j] for i,j in combinations(refentry.values, 2)]

                # and return the 10th percentile (novelty) and median (conventionality)
                return pd.Series([np.percentile(ref_pair_zscores, 10), np.median(ref_pair_zscores)], 
                                 index=['NoveltyScore', 'ConventionalityScore'])
            else:
                return pd.Series([None, None], 
                                 index=['NoveltyScore', 'ConventionalityScore'])

        with np.errstate(divide='ignore',invalid='ignore'):
            ynovelty_score = ypub2ref.groupby('CitingPublicationInt')['CitedJournalInt'].apply(_ref_distribution)

        # finally do some reordering to make the returned dataframe nice
        ynovelty_score = ynovelty_score.unstack(1).reset_index().rename(columns={'CitingPublicationInt':'PublicationId'})
        ynovelty_score['PublicationId'] = [int2pub.get(i, None) for i in ynovelty_score['PublicationId'].values]
        if not focus_pub_ids is None:
            ynovelty_score = ynovelty_score.loc[isin_sorted(ynovelty_score['PublicationId'].values, focus_pub_ids)]
        
        novelty_score.append(ynovelty_score)

    novelty_score = pd.concat(novelty_score)

    return novelty_score
