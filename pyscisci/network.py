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

def threshold_network(adj_mat, threshold=0):
    """

    """
    if adj_mat.getformat() != 'coo':
        adj_mat = spsparse.coo_matrix(adj_mat)

    adj_mat.data[adj_mat.data <=threshold] = 0
    adj_mat.eliminate_zeros()

    return adj_mat

def largest_connected_component_vertices(adj_mat):
    """
    """
    n_components, labels = spsparse.csgraph.connected_components(adj_mat)
    comidx, compsizes = np.unique(labels, return_counts=True)

    return np.arange(adj_mat.shape[0])[labels==np.argmax(compsizes)]

def coauthorship_network(paa_df, focus_author_ids=None, focus_constraint='authors', temporal=False, show_progress=False):
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

    :param temporal : bool, default False
        If True, compute the adjacency matrix using only publications for each year.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.


    Returns
    -------
    coo_matrix or dict of coo_matrix
        If temporal == False:
            The adjacency matrix for the co-authorship network

        If temporal == True:
            A dictionary with key for each year, and value of the adjacency matrix for the co-authorship network induced by publications in that year.

    author2int, dict
        A mapping of AuthorIds to the row/column of the adjacency matrix.

    """
    required_columns = ['AuthorId', 'PublicationId']
    if temporal:
        required_columns.append('Year')
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

    paa_df.drop_duplicates(subset=['AuthorId', 'PublicationId'], inplace=True)

    #  map authors to the rows of the bipartite adj mat
    author2int = {aid:i for i, aid in enumerate(np.sort(paa_df['AuthorId'].unique()))}
    Nauthors = paa_df['AuthorId'].nunique()

    paa_df['AuthorId'] = [author2int[aid] for aid in paa_df['AuthorId'].values]

    #  map publications to the columns of the bipartite adj mat
    pub2int = {pid:i for i, pid in enumerate(np.sort(paa_df['PublicationId'].unique()))}
    Npubs = paa_df['PublicationId'].nunique()

    paa_df['PublicationId'] = [pub2int[pid] for pid in paa_df['PublicationId'].values]

    if temporal:
        years = np.sort(paa_df['Year'].unique())

        temporal_adj = {}
        for y in years:
            bipartite_adj = dataframe2bipartite(paa_df.loc[paa_df['Year'] == y], 'AuthorId', 'PublicationId', (Nauthors, Npubs) )

            adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

            # remove diagonal entries
            adj_mat.setdiag(0)
            adj_mat.eliminate_zeros()

            temporal_adj[y] = adj_mat

        return temporal_adj, author2int

    else:
        bipartite_adj = dataframe2bipartite(paa_df, 'AuthorId', 'PublicationId', (Nauthors, Npubs) )

        adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

        # remove diagonal entries
        adj_mat.setdiag(0)
        adj_mat.eliminate_zeros()

        return adj_mat, author2int


def cogroupby(df, N):
    adj_mat = spsparse.dok_matrix( (N,N), dtype=int)
    def inducedcombos(authorlist):
        if authorlist.shape[0] >= 2:
            for i,j in combinations(authorlist, 2):
                adj_mat[i,j] += 1

    tqdm.pandas(desc='CoAuthorship')
    df.groupby('PublicationId')['AuthorId'].progress_apply(inducedcombos)

    adj_mat = adj_mat + adj_mat.T

    return adj_mat


def dataframe2bipartite(df, rowname, colname, shape=None, weightname=None):

    if shape is None:
        shape = (int(df[rowname].max()+1), int(df[colname].max()+1) )

    if weightname is None:
        weights = np.ones(df.shape[0], dtype=int)
    else:
        weights = df[weightname].values

    # create a bipartite adj matrix connecting authors to their publications
    bipartite_adj = spsparse.coo_matrix( ( weights,
                                        (df[rowname].values, df[colname].values) ),
                                        shape=shape, dtype=weights.dtype)

    bipartite_adj.sum_duplicates()

    return bipartite_adj

def project_bipartite_mat(bipartite_adj, project_to = 'row'):

    if project_to == 'row':
        adj_mat = bipartite_adj.dot(bipartite_adj.T).tocoo()
    elif project_to == 'col':
        adj_mat = bipartite_adj.T.dot(bipartite_adj).tocoo()

    return adj_mat

def extract_multiscale_backbone(Xs, alpha):
    """
    A sparse matrix implemntation of the multiscale backbone.

    References
    ----------
    Serrano et al. (2009) Extracting the multiscale backbone of complex weighted networks.  PNAS.

    Parameters
    ----------
    :param Xs : numpy.array or sp.sparse matrix
        The adjacency matrix for the network.

    :param alpha : float
        The significance value.


    Returns
    -------
    coo_matrix
        The directed, weighted multiscale backbone

    """

    X = spsparse.coo_matrix(Xs)
    X.eliminate_zeros()

    #normalize
    row_sums = X.sum(axis = 1)
    degrees = X.getnnz(axis = 1)


    pijs = np.multiply(X.data, 1.0/np.array(row_sums[X.row]).squeeze())
    powers = degrees[X.row.squeeze()] - 1

    # equation 2 => where 1 - (k - 1) * integrate.quad(lambda x: (1 - x) ** (k - 2)) = (1-x)**(k - 1) if k  > 1
    significance = np.logical_and(pijs < 1, np.power(1.0 - pijs, powers) < alpha)

    keep_graph = spsparse.coo_matrix((X.data[significance], (X.row[significance], X.col[significance])), shape = X.shape)
    keep_graph.eliminate_zeros()

    return keep_graph


def cocitation_network(pub2ref_df, focus_pub_ids=None, focus_constraint='citing', temporal=False, show_progress=False):
    """
    Create the co-citation network.

    Parameters
    ----------
    :param pub2ref_df : DataFrame
        A DataFrame with the links between authors and publications.

    :param focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to seed the cocitation-network.

    :param focus_constraint : str, default `citing`
        If focus_author_ids is not None:
            `citing` : the `focus_pub_ids' defines the citation set, giving only the co-citations between the references
                of the publications from this set.
            `cited` : the `focus_pub_ids' defines the cocitation node set.
            'egocited' : the `focus_pub_ids' defines a seed set, such that all other publications must have been co-citeed with
                at least one publication from this set.

    :param temporal : bool, default False
        If True, compute the adjacency matrix using only publications for each year.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.


    Returns
    -------
    coo_matrix or dict of coo_matrix
        If temporal == False:
            The adjacency matrix for the co-citation network

        If temporal == True:
            A dictionary with key for each year, and value of the adjacency matrix for the cocitation network induced
            by citing publications in that year.

    pub2int, dict
        A mapping of PublicationIds to the row/column of the adjacency matrix.

    """
    required_columns = ['CitedPublicationId', 'CitingPublicationId']
    if temporal:
        required_columns.append('CitingYear')
    check4columns(pub2ref_df, required_columns)
    pub2ref_df = pub2ref_df[required_columns].dropna()

    if not focus_pub_ids is None:
        focus_pub_ids = np.sort(focus_pub_ids)

        # identify the subset of the publications we need to form the network
        if focus_constraint == 'citing':
            # take only the links that have a citing publication from the `focus_pub_ids'
            pub2ref_df = pub2ref_df.loc[isin_sorted(pub2ref_df['CitingPublicationId'].values, focus_pub_ids)]

        elif focus_constraint == 'cited':
            # take only the links that have a cited publication from the `focus_pub_ids'
            pub2ref_df = pub2ref_df.loc[isin_sorted(pub2ref_df['CitedPublicationId'].values, focus_pub_ids)]

        elif focus_constraint == 'egocited':
            # take all publications that cite one of the publications in `focus_pub_ids'
            focus_citing_pubs = np.sort(pub2ref_df.loc[isin_sorted(pub2ref_df['CitedPublicationId'].values, focus_pub_ids)]['CitingPublicationId'].unique())
            # then take all the links that have a citing publication from the `focus_citing_pubs'
            pub2ref_df = pub2ref_df.loc[isin_sorted(pub2ref_df['CitingPublicationId'].values, focus_citing_pubs)]
            del focus_citing_pubs

    pub2ref_df.drop_duplicates(subset=['CitingPublicationId', 'CitedPublicationId'], inplace=True)

    if pub2ref_df.shape[0] > 0:
        #  map cited publications to the rows of the bipartite adj mat
        cited2int = {pid:i for i, pid in enumerate(np.sort(pub2ref_df['CitedPublicationId'].unique()))}
        Ncited = pub2ref_df['CitedPublicationId'].nunique()

        pub2ref_df['CitedPublicationId'] = [cited2int[pid] for pid in pub2ref_df['CitedPublicationId'].values]

        #  map citing publications to the columns of the bipartite adj mat
        citing2int = {pid:i for i, pid in enumerate(np.sort(pub2ref_df['CitingPublicationId'].unique()))}
        Nciting = pub2ref_df['CitingPublicationId'].nunique()

        pub2ref_df['CitingPublicationId'] = [citing2int[pid] for pid in pub2ref_df['CitingPublicationId'].values]

        if temporal:
            years = np.sort(pub2ref_df['CitingYear'].unique())

            temporal_adj = {}
            for y in years:
                bipartite_adj = dataframe2bipartite(pub2ref_df.loc[pub2ref_df['CitingYear'] == y], 'CitedPublicationId', 'CitingPublicationId', (Ncited, Nciting) )

                adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

                # remove diagonal entries
                adj_mat.setdiag(0)
                adj_mat.eliminate_zeros()

                temporal_adj[y] = adj_mat

            return temporal_adj, cited2int

        else:
            bipartite_adj = dataframe2bipartite(pub2ref_df, 'CitedPublicationId', 'CitingPublicationId', (Ncited, Nciting) )

            adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

            # remove diagonal entries
            adj_mat.setdiag(0)
            adj_mat.eliminate_zeros()

            return adj_mat, cited2int

    else:
        return spsparse.coo_matrix(), {}

def cociting_network(pub2ref_df, focus_pub_ids=None, focus_constraint='citing', temporal=False, show_progress=False):
    """
    Create the co-citing network.  Each node is a publication, two publications are linked if they cite the same article.

    Parameters
    ----------
    :param pub2ref_df : DataFrame
        A DataFrame with the links between authors and publications.

    :param focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to seed the cocitation-network.

    :param focus_constraint : str, default `citing`
        If focus_author_ids is not None:
            `citing` : the `focus_pub_ids' defines the citation set, giving only the co-citations between the references
                of the publications from this set.
            `cited` : the `focus_pub_ids' defines the cocitation node set.

    :param show_progress : bool, default False
        If True, show a progress bar tracking the calculation.


    Returns
    -------
    coo_matrix or dict of coo_matrix
        The adjacency matrix for the co-citing network

    pub2int, dict
        A mapping of PublicationIds to the row/column of the adjacency matrix.

    """
    required_columns = ['CitedPublicationId', 'CitingPublicationId']
    check4columns(pub2ref_df, required_columns)
    pub2ref_df = pub2ref_df[required_columns].dropna()

    if not focus_pub_ids is None:
        focus_pub_ids = np.sort(focus_pub_ids)

        # identify the subset of the publications we need to form the network
        if focus_constraint == 'citing':
            # take only the links that have a citing publication from the `focus_pub_ids'
            pub2ref_df = pub2ref_df.loc[isin_sorted(pub2ref_df['CitingPublicationId'].values, focus_pub_ids)]

        elif focus_constraint == 'cited':
            # take only the links that have a cited publication from the `focus_pub_ids'
            pub2ref_df = pub2ref_df.loc[isin_sorted(pub2ref_df['CitedPublicationId'].values, focus_pub_ids)]

    pub2ref_df.drop_duplicates(subset=['CitingPublicationId', 'CitedPublicationId'], inplace=True)

    if pub2ref_df.shape[0] > 0:
        #  map cited publications to the rows of the bipartite adj mat
        cited2int = {pid:i for i, pid in enumerate(np.sort(pub2ref_df['CitedPublicationId'].unique()))}
        Ncited = pub2ref_df['CitedPublicationId'].nunique()

        pub2ref_df['CitedPublicationId'] = [cited2int[pid] for pid in pub2ref_df['CitedPublicationId'].values]

        #  map citing publications to the columns of the bipartite adj mat
        citing2int = {pid:i for i, pid in enumerate(np.sort(pub2ref_df['CitingPublicationId'].unique()))}
        Nciting = pub2ref_df['CitingPublicationId'].nunique()

        pub2ref_df['CitingPublicationId'] = [citing2int[pid] for pid in pub2ref_df['CitingPublicationId'].values]

        bipartite_adj = dataframe2bipartite(pub2ref_df, 'CitedPublicationId', 'CitingPublicationId', (Ncited, Nciting) )

        adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'col')

        # remove diagonal entries
        adj_mat.setdiag(0)
        adj_mat.eliminate_zeros()

        return adj_mat, cited2int

    else:
        return spsparse.coo_matrix(), {}

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

