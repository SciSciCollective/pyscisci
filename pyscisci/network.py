# -*- coding: utf-8 -*-
"""
.. module:: bibnetwork
    :synopsis: The main Network class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import pandas as pd

import scipy.sparse as spsparse

from pyscisci.utils import isin_sorted, zip2dict, check4columns

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def coauthorship_network(paa_df, focus_author_ids=None, focus_constraint='authors', show_progress=False):
    """
    Create the co-authorship network.

    Parameters
    ----------
    :param paa_df : DataFrame
        A DataFrame with the links between authors and publications.

    :param focus_author_ids : numpy array or list, default None
        A list of the AuthorIds to seed the coauthorship-network.

    :param focus_constraint : str, default `authors`
        If focus_author_ids is not None:
            `authors` : the `focus_author_ids' defines the node set, giving only the co-authorships between authors in the set.
            `publications` : the publication history of `focus_author_ids' defines the edge set, giving the co-authorhips where at least 
                                one author from `focus_author_ids' was involved.
            'ego' : the `focus_author_ids' defines a seed set, such that all authors must have co-authored at least one publication with 
                                an author from `focus_author_ids', but co-authorships are also found between the second-order author sets. 

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.


    Returns
    -------
    coo_matrix
        The adjacency matrix for the co-authorship network

    author2int, dict
        A mapping of AuthorIds to the row/column of the adjacency matrix.

    """
    required_columns = ['AuthorId', 'PublicationId']
    check4columns(paa_df, required_columns)
    paa_df = paa_df[required_columns].dropna()

    if not focus_author_ids is None:
        focus_author_ids = np.sort(focus_author_ids)
        
        # identify the subset of the publications we need to form the network
        if focus_constraint == 'authors':
            # take only the publication-author links that have an author from the `focus_author_ids'
            paa_df = paa_df.loc[isin_sorted(paa_df['AuthorId'].values, focus_author_ids)]
        
        elif focus_constraint == 'publications':
            # take all publications authored by an author from the `focus_author_ids'
            focus_pubs = np.sort(paa_df.loc[isin_sorted(paa_df['AuthorId'].values, focus_author_ids)]['PublicationId'].unique())
            # then take only the subset of publication-author links inducded by these publications
            paa_df = paa_df.loc[isin_sorted(paa_df['PublicationId'].values, focus_pubs)]
            del focus_pubs

        elif focus_constraint == 'ego':
            # take all publications authored by an author from the `focus_author_ids'
            focus_pubs = np.sort(paa_df.loc[isin_sorted(paa_df['AuthorId'].values, focus_author_ids)]['PublicationId'].unique())
            # then take all authors who contribute to this subset of publications
            focus_author_ids = np.sort(paa_df.loc[isin_sorted(paa_df['PublicationId'].values, focus_pubs)]['AuthorId'].unique())
            del focus_pubs
            # finally take the publication-author links that have an author from the above ego subset
            paa_df = paa_df.loc[isin_sorted(paa_df['AuthorId'].values, focus_author_ids)]

    #  map authors to the rows of the bipartite adj mat
    author2int = {aid:i for i, aid in enumerate(np.sort(paa_df['AuthorId'].unique()))}
    Nauthors = paa_df['AuthorId'].nunique()

    paa_df['AuthorId'] = [author2int[aid] for aid in paa_df['AuthorId'].values]

    #  map publications to the columns of the bipartite adj mat
    pub2int = {pid:i for i, pid in enumerate(np.sort(paa_df['PublicationId'].unique()))}
    Npubs = paa_df['PublicationId'].nunique()

    paa_df['PublicationId'] = [pub2int[pid] for pid in paa_df['PublicationId'].values]
    
    # create a bipartite adj matrix connecting authors to their publications
    bipartite_adj = spsparse.coo_matrix( ( np.ones(paa_df.shape[0], dtype=int), 
                                        (paa_df['AuthorId'].values, paa_df['PublicationId'].values) ),
                                        shape=(Nauthors, Npubs), dtype=int)

    bipartite_adj.sum_duplicates()

    # now project the bipartite adj matrix onto the authors 
    adj_mat = bipartite_adj.dot(bipartite_adj.T).tocoo()

    # remove diagonal entries
    adj_mat.setdiag(0)
    adj_mat.eliminate_zeros()
    
    return adj_mat, author2int



def cocited_edgedict(refdf):

    cocite_dict = defaultdict(int)
    def count_cocite(refseries):
        for i, j in combinations(np.sort(refseries.values), 2):
            cocite_dict[(i, j)] += 1

    refdf.groupby('CitingPublicationId', sort=False)['CitedPublicationId'].apply(count_cocite)

    return cocite_dict

def temporal_cocited_edgedict(pub2ref, pub2year):

    required_pub2ref_columns = ['CitingPublicationId', 'CitedPublicationId']
    check4columns(pub2ref, required_pub2ref_columns)
    pub2ref = pub2ref[required_pub2ref_columns]

    year_values = sorted(list(set(pub2year.values())))

    # we need the citation counts and cocitation network
    temporal_cocitation_dict = {y:defaultdict(set) for y in year_values}
    temporal_citation_dict = {y:defaultdict(int) for y in year_values}

    def count_cocite(cited_df):
        y = pub2year[cited_df.name]

        for citedpid in cited_df['CitedPublicationId'].values:
            temporal_citation_dict[y][citedpid] += 1
        for icitedpid, jcitedpid in combinations(cited_df['CitedPublicationId'].values, 2):
            temporal_cocitation_dict[y][icitedpid].add(jcitedpid)
            temporal_cocitation_dict[y][jcitedpid].add(icitedpid)

    pub2ref.groupby('CitingPublicationId', sort=False).apply(count_cocite)


def dataframe2sparse(df, index_columns = ['name_citing', 'name_cited'], weight_values = None):

    """
    Take a pandas Data Frame as an edge list and convert it into a sparse adjacency matrix
    """

    # convert values 2 int
    nameindex = pd.DataFrame(pd.unique(df[index_columns].values.ravel()), columns = ['name'])
    nameindex.sort_values('name', inplace = True)
    nameindex.reset_index(inplace = True, drop = True)
    nameindex['name2int'] = nameindex.index.values

    # sparse matrix size
    N = nameindex.shape[0]

    df = pd.merge(df, nameindex, left_on = 'name_citing', right_on = 'name',
                   copy = 'False', how = 'left', suffixes = ['', '_citing'])
    del df['name']

    df = pd.merge(df, nameindex, left_on = 'name_cited', right_on = 'name',
                   copy = 'False', how = 'left', suffixes = ['_citing', '_cited'])
    del df['name']

    if weight_values is None:
        weight_values = np.ones(df.shape[0])
    else:
        weight_values = df[weight_values].values

    mat = sp.coo_matrix((weight_values, (df['name2int_citing'].values, df['name2int_cited'].values)), shape = (N,N))
    mat.sum_duplicates()

    return mat, nameindex


def create_citation_edgelist(database, publication_subset = [], temporal = True):

    if len(publication_subset) == 0:
        publication_subset = database.publication_ids()

    edgelist = [[pubid, refid] for pubid in publication_subset for refid in database.get_pub(pubid).references]
    return edgelist

