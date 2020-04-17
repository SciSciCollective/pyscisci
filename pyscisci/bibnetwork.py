# -*- coding: utf-8 -*-
"""
.. module:: bibnetwork
    :synopsis: The main Network class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import numpy as np
import pandas as pd
import igraph
import scipy.sparse as sparse


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

