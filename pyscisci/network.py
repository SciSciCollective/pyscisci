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

from pyscisci.utils import isin_sorted, zip2dict, check4columns, value_to_int
from pyscisci.sparsenetworkutils import threshold_network, largest_connected_component_vertices, dataframe2bipartite, project_bipartite_mat

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def coauthorship_network(paa, focus_author_ids=None, focus_constraint='authors', temporal=False, show_progress=False):
    """
    Create the co-authorship network.

    Parameters
    ----------
    paa : DataFrame
        A DataFrame with the links between authors and publications.

    focus_author_ids : numpy array or list, default None
        A list of the AuthorIds to seed the coauthorship-network.

    focus_constraint : str, default 'authors'
        If focus_author_ids is not None:
            - 'authors' : the 'focus_author_ids' defines the node set, giving only the co-authorships between authors in the set.
            - 'publications' : the publication history of `focus_author_ids' defines the edge set, giving the co-authorhips where at least one author from `focus_author_ids' was involved.
            - 'ego' : the 'focus_author_ids' defines a seed set, such that all authors must have co-authored at least one publication with an author from `focus_author_ids', but co-authorships are also found between the second-order author sets.

    temporal : bool, default False
        If True, compute the adjacency matrix using only publications for each year.

    show_progress : bool, default False
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


    |
    

    """
    required_columns = ['AuthorId', 'PublicationId']
    if temporal:
        required_columns.append('Year')
    check4columns(paa, required_columns)
    paa = paa[required_columns].dropna()

    if not focus_author_ids is None:
        focus_author_ids = np.sort(focus_author_ids)

        # identify the subset of the publications we need to form the network
        if focus_constraint == 'authors':
            # take only the publication-author links that have an author from the `focus_author_ids'
            paa = paa.loc[isin_sorted(paa['AuthorId'].values, focus_author_ids)]

        elif focus_constraint == 'publications':
            # take all publications authored by an author from the `focus_author_ids'
            focus_pubs = np.sort(paa.loc[isin_sorted(paa['AuthorId'].values, focus_author_ids)]['PublicationId'].unique())
            # then take only the subset of publication-author links inducded by these publications
            paa = paa.loc[isin_sorted(paa['PublicationId'].values, focus_pubs)]
            del focus_pubs

        elif focus_constraint == 'ego':
            # take all publications authored by an author from the `focus_author_ids'
            focus_pubs = np.sort(paa.loc[isin_sorted(paa['AuthorId'].values, focus_author_ids)]['PublicationId'].unique())
            # then take all authors who contribute to this subset of publications
            focus_author_ids = np.sort(paa.loc[isin_sorted(paa['PublicationId'].values, focus_pubs)]['AuthorId'].unique())
            del focus_pubs
            # finally take the publication-author links that have an author from the above ego subset
            paa = paa.loc[isin_sorted(paa['AuthorId'].values, focus_author_ids)]

    paa.drop_duplicates(subset=['AuthorId', 'PublicationId'], inplace=True)

    #  map authors to the rows of the bipartite adj mat
    author2int = {aid:i for i, aid in enumerate(np.sort(paa['AuthorId'].unique()))}
    Nauthors = paa['AuthorId'].nunique()

    paa['AuthorId'] = [author2int[aid] for aid in paa['AuthorId'].values]

    #  map publications to the columns of the bipartite adj mat
    pub2int = {pid:i for i, pid in enumerate(np.sort(paa['PublicationId'].unique()))}
    Npubs = paa['PublicationId'].nunique()

    paa['PublicationId'] = [pub2int[pid] for pid in paa['PublicationId'].values]

    if temporal:
        years = np.sort(paa['Year'].unique())

        temporal_adj = {}
        for y in years:
            bipartite_adj = dataframe2bipartite(paa.loc[paa['Year'] == y], 'AuthorId', 'PublicationId', (Nauthors, Npubs) )

            adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

            # remove diagonal entries
            adj_mat.setdiag(0)
            adj_mat.eliminate_zeros()

            temporal_adj[y] = adj_mat

        return temporal_adj, author2int

    else:
        bipartite_adj = dataframe2bipartite(paa, 'AuthorId', 'PublicationId', (Nauthors, Npubs) )

        adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

        # remove diagonal entries
        adj_mat.setdiag(0)
        adj_mat.eliminate_zeros()

        return adj_mat, author2int


def extract_multiscale_backbone(Xs, alpha):
    """
    A sparse matrix implemntation of the multiscale backbone.

    References
    ----------
    Serrano et al. (2009) Extracting the multiscale backbone of complex weighted networks.  PNAS.

    Parameters
    ----------
    Xs : numpy.array or sp.sparse matrix
        The adjacency matrix for the network.

    alpha : float
        The significance value.


    Returns
    -------
    coo_matrix
        The directed, weighted multiscale backbone



    |
    

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


def cocitation_network(pub2ref, focus_pub_ids=None, focus_constraint='citing', cited_col_name = 'CitedPublicationId', 
    citing_col_name = 'CitingPublicationId', temporal=False, show_progress=False):
    """
    Create the co-citation network.


    Parameters
    ----------
    pub2ref : DataFrame
        A DataFrame with the links between authors and publications.

    focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to seed the cocitation-network.

    focus_constraint : str, default 'citing'
        If focus_author_ids is not None
            -'citing' : the 'focus_pub_ids' defines the citation set, giving only the co-citations between the references
                of the publications from this set.
            -'cited' : the 'focus_pub_ids' defines the cocitation node set.
            -'egocited' : the 'focus_pub_ids' defines a seed set, such that all other publications must have been co-citeed with
                at least one publication from this set.
    
    cited_col_name : str, default 'CitedPublicationId'
        The name of the cited value column in the DataFrame pub2ref
    
    citing_col_name : str, default 'CitingPublicationId'
        The name of the citing value column in the DataFrame pub2ref

    temporal : bool, default False
        If True, compute the adjacency matrix using only publications for each year.

    show_progress : bool, default False
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
    required_columns = [cited_col_name, citing_col_name]
    if temporal:
        required_columns.append('CitingYear')
    check4columns(pub2ref, required_columns)
    pub2ref = pub2ref[required_columns].dropna()

    if not focus_pub_ids is None:
        focus_pub_ids = np.sort(focus_pub_ids)

        # identify the subset of the publications we need to form the network
        if focus_constraint == 'citing':
            # take only the links that have a citing publication from the `focus_pub_ids'
            pub2ref = pub2ref.loc[isin_sorted(pub2ref[citing_col_name].values, focus_pub_ids)]

        elif focus_constraint == 'cited':
            # take only the links that have a cited publication from the `focus_pub_ids'
            pub2ref = pub2ref.loc[isin_sorted(pub2ref[cited_col_name].values, focus_pub_ids)]

        elif focus_constraint == 'egocited':
            # take all publications that cite one of the publications in `focus_pub_ids'
            focus_citing_pubs = np.sort(pub2ref.loc[isin_sorted(pub2ref[cited_col_name].values, focus_pub_ids)][citing_col_name].unique())
            # then take all the links that have a citing publication from the `focus_citing_pubs'
            pub2ref = pub2ref.loc[isin_sorted(pub2ref[citing_col_name].values, focus_citing_pubs)]
            del focus_citing_pubs

    pub2ref.drop_duplicates(subset=[citing_col_name, cited_col_name], inplace=True)

    if pub2ref.shape[0] > 0:
        #  map cited publications to the rows of the bipartite adj mat
        pub2ref[cited_col_name], cited2int = value_to_int(pub2ref[cited_col_name].values, sort_values='value', return_map=True)
        Ncited = len(cited2int)

        #  map citing publications to the columns of the bipartite adj mat
        pub2ref[citing_col_name], citing2int = value_to_int(pub2ref[citing_col_name].values, sort_values='value', return_map=True)
        Nciting = len(citing2int)

        if temporal:
            years = np.sort(pub2ref['CitingYear'].unique())

            temporal_adj = {}
            for y in years:
                bipartite_adj = dataframe2bipartite(pub2ref.loc[pub2ref['CitingYear'] == y], cited_col_name, citing_col_name, (Ncited, Nciting) )

                adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

                # remove diagonal entries
                adj_mat.setdiag(0)
                adj_mat.eliminate_zeros()

                temporal_adj[y] = adj_mat

            return temporal_adj, cited2int

        else:
            bipartite_adj = dataframe2bipartite(pub2ref, cited_col_name, citing_col_name, (Ncited, Nciting) )

            adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'row')

            # remove diagonal entries
            adj_mat.setdiag(0)
            adj_mat.eliminate_zeros()

            return adj_mat, cited2int

    else:
        return spsparse.coo_matrix(), {}

def cociting_network(pub2ref, focus_pub_ids=None, focus_constraint='citing', cited_col_name = 'CitedPublicationId', 
    citing_col_name = 'CitingPublicationId', temporal=False, show_progress=False):
    """
    Create the co-citing network.  Each node is a publication, two publications are linked if they cite the same article.


    Parameters
    ----------

    pub2ref : DataFrame
        A DataFrame with the links between authors and publications.

    focus_pub_ids : numpy array or list, default None
        A list of the PublicationIds to seed the cocitation-network.

    focus_constraint : str, default 'citing'
        If focus_author_ids is not None
            - 'citing' : the 'focus_pub_ids' defines the citation set, giving only the co-citations between the references
                of the publications from this set.
            - 'cited' : the 'focus_pub_ids' defines the cocitation node set.

    cited_col_name : str, default 'CitedPublicationId'
        The name of the cited value column in the DataFrame pub2ref
    
    citing_col_name : str, default 'CitingPublicationId'
        The name of the citing value column in the DataFrame pub2ref

    show_progress : bool, default False
        If True, show a progress bar tracking the calculation.


    Returns
    -------

    coo_matrix or dict of coo_matrix
        The adjacency matrix for the co-citing network

    pub2int, dict
        A mapping of PublicationIds to the row/column of the adjacency matrix.



    |
    

    """
    required_columns = [cited_col_name, citing_col_name]
    check4columns(pub2ref, required_columns)
    pub2ref = pub2ref[required_columns].dropna()

    if not focus_pub_ids is None:
        focus_pub_ids = np.sort(focus_pub_ids)

        # identify the subset of the publications we need to form the network
        if focus_constraint == 'citing':
            # take only the links that have a citing publication from the `focus_pub_ids'
            pub2ref = pub2ref.loc[isin_sorted(pub2ref[citing_col_name].values, focus_pub_ids)]

        elif focus_constraint == 'cited':
            # take only the links that have a cited publication from the `focus_pub_ids'
            pub2ref = pub2ref.loc[isin_sorted(pub2ref[cited_col_name].values, focus_pub_ids)]

    pub2ref.drop_duplicates(subset=[citing_col_name, cited_col_name], inplace=True)

    if pub2ref.shape[0] > 0:
        #  map cited publications to the rows of the bipartite adj mat
        pub2ref[cited_col_name], cited2int = value_to_int(pub2ref[cited_col_name].values, sort_values='value', return_map=True)
        Ncited = len(cited2int)

        #  map citing publications to the columns of the bipartite adj mat
        pub2ref[citing_col_name], citing2int = value_to_int(pub2ref[citing_col_name].values, sort_values='value', return_map=True)
        Nciting = len(citing2int)

        bipartite_adj = dataframe2bipartite(pub2ref, cited_col_name, citing_col_name, (Ncited, Nciting) )

        adj_mat = project_bipartite_mat(bipartite_adj, project_to = 'col')

        # remove diagonal entries
        adj_mat.setdiag(0)
        adj_mat.eliminate_zeros()

        return adj_mat, citing2int

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

    def count_cocite(cited):
        y = pub2year[cited.name]

        for citedpid in cited['CitedPublicationId'].values:
            temporal_citation_dict[y][citedpid] += 1
        for icitedpid, jcitedpid in combinations(cited['CitedPublicationId'].values, 2):
            temporal_cocitation_dict[y][icitedpid].add(jcitedpid)
            temporal_cocitation_dict[y][jcitedpid].add(icitedpid)

    pub2ref.groupby('CitingPublicationId', sort=False).apply(count_cocite)


def create_citation_edgelist(database, publication_subset = [], temporal = True):

    if len(publication_subset) == 0:
        publication_subset = database.publication_ids()

    edgelist = [[pubid, refid] for pubid in publication_subset for refid in database.get_pub(pubid).references]
    return edgelist

def estimate_resolution(G, com):
    """
    Newman, MEJ (2016) Community detection in networks: Modularity optimization and maximum likelihood are equivalent. Phy. Rev. E
    """
    m = G.number_of_edges()
    
    # eq 16
    kappas = [sum(deg for n, deg in G.degree(c)) for c in com.communities]
    m_in = [G.subgraph(c).number_of_edges() for c in com.communities]
    
    denom = np.sum(np.square(kappas)) / (2*m)
    
    # eq 17
    omega_in = 2*sum(m_in) / denom
    
    # eq 18
    omega_out = (2*m - 2*sum(m_in)) / (2*m - denom)
    
    # eq 15
    gamma = (omega_in - omega_out) / (np.log(omega_in) - np.log(omega_out))
    
    return gamma

