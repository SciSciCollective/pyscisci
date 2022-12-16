# -*- coding: utf-8 -*-
"""
.. module:: sparsenetworkutils
    :synopsis: The main Network class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import sys
import numpy as np
import pandas as pd

from scipy import repeat, sqrt, where, square, absolute
import scipy.sparse as spsparse

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
    A sparse matrix implemntation of the multiscale backbone :cite:`Serrano2009backbone`.

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

def sparse_pagerank_scipy(adjmat, alpha=0.85, personalization=None, initialization=None,
                   max_iter=100, tol=1.0e-6, dangling=None):
    
    """
    Pagerank for sparse matrices using the power method
    """

    N, _ = adjmat.shape
    assert(adjmat.shape==(N,N))
    
    if N == 0:
        return np.array([])

    out_strength = np.array(adjmat.sum(axis=1)).flatten()
    out_strength[out_strength != 0] = 1.0 / out_strength[out_strength != 0]
    
    Q = spsparse.spdiags(out_strength.T, 0, *adjmat.shape, format='csr')
    adjmat = Q * adjmat

    # initial vector
    if initialization is None:
        x = repeat(1.0 / N, N)
    
    else:
        x = initialization / initialization.sum()

    # Personalization vector
    if personalization is None:
        p = repeat(1.0 / N, N)
    else:
        p = personalization / personalization.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        dangling_weights = dangling / dangling.sum()

    is_dangling = where(out_strength == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * adjmat + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        err = absolute(x - xlast).sum()
        if err < N * tol:
            return x

    print('power iteration failed to converge in %d iterations.' % max_iter)


def sparse_eigenvector_centrality_scipy(adjmat, max_iter=100, tol=1.0e-6, initialization=None):

    adjmat = spsparse.csr_matrix(adjmat)

    N, _ = adjmat.shape
    assert(adjmat.shape==(N,N))
    
    if N == 0:
        return np.array([])

    # initial vector
    if initialization is None:
        x = repeat(1.0 / N, N)
    
    else:
        x = initialization / initialization.sum()

    
    # make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = xlast.copy()  # Start with xlast times I to iterate with (A+I)
        # do the multiplication y^T = x^T A (left eigenvector)
        x = xlast*adjmat

        
        norm = sqrt( square(x).sum() ) or 1
        x = x/norm

        # Check for convergence (in the L_1 norm).
        err = absolute(x - xlast).sum()
        if err < N * tol:
            return x
    print('power iteration failed to converge in %d iterations.' % max_iter)
