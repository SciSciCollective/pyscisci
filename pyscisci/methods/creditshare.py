# -*- coding: utf-8 -*-
"""
.. module:: credit sharing
    :synopsis: Set of functions for calcuating credit share amongst authors.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import pandas as pd
import numpy as np

from pyscisci.utils import isin_sorted, groupby_count
from pyscisci.network import cocitation_network

def credit_share(focus_pid, pub2ref, pub2author, temporal=False, normed=False, show_progress=False):
    """
    Calculate the credit share for each author of a publication based on :cite:`Shen2014credit`.

    Parameters
    ----------
    :param focus_pid : int, str
        The focus publication id.

    :param pub2ref : DataFrame
        A DataFrame with the citation information for each Publication.

    :param pub2author : DataFrame
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

    # the focus publication's authors
    focus_authors = np.sort(pub2author.loc[pub2author['PublicationId']==focus_pid]['AuthorId'].unique())
    author2int = {aid:i for i, aid in enumerate(focus_authors)}

    if focus_authors.shape[0] > 1:
        # start by getting the co-citation network around the focus publication
        adj_mat, cited2int = cocitation_network(pub2ref, focus_pub_ids=np.sort([focus_pid]), focus_constraint='egocited',
                temporal=temporal, show_progress=show_progress)

        # get the authorships for the publications in the cocitation network
        cocited_pubs = np.sort(list(cited2int.keys()))
        pa = pub2author.loc[isin_sorted(pub2author['PublicationId'].values, cocited_pubs)]

        if cocited_pubs.shape[0] > 0:
            # the credit allocation matrix has a row for each focus author, and a column for each cocited publication (including the focus pub)
            credit_allocation_mat = np.zeros((focus_authors.shape[0], cocited_pubs.shape[0]), dtype = float)

            # for each cocited publication, we count the number of authors
            # and assign to each focus author, their fractional share of the credit (1 divided by the number of authors)
            for cocitedid, adf in pa.groupby('PublicationId'):
                author2row = [author2int[aid] for aid in adf['AuthorId'].unique() if not author2int.get(aid, None) is None]
                if len(author2row) > 0:
                    credit_allocation_mat[author2row, cited2int[cocitedid]] = 1.0/adf['AuthorId'].nunique()

            if temporal:
                # temporal credit allocation - broken down by year

                # we need the temporal citations to the focus article
                focus_citations = groupby_count(pub2ref.loc[isin_sorted(pub2ref['CitedPublicationId'].values, np.sort([focus_pid]))],
                    colgroupby='CitingYear', colcountby='CitingPublicationId', count_unique=True, show_progress=False)
                focus_citations={y:c for y,c in focus_citations[['CitingYear', 'CitingPublicationIdCount']].values}

                # when temporal is True, a temporal adj mat is returned where each key is the year
                years = np.sort(list(adj_mat.keys()))

                cocite_counts = np.zeros((years.shape[0], cocited_pubs.shape[0]), dtype=float)

                for iy, y in enumerate(years):
                    cocite_counts[iy] = adj_mat[y].tocsr()[cited2int[focus_pid]].todense()#set the off-diagonal to be the total co-citations from that year
                    cocite_counts[iy, cited2int[focus_pid]] = focus_citations[y]          #set the diagonal to be the total citations from that year

                cocite_counts = cocite_counts.cumsum(axis=0)

            else:
                # just do credit allocation with the full cocitation matrix
                cocite_counts = adj_mat.tocsr()[cited2int[focus_pid]].todense()

                # the co-citation matrix misses the number of citations to the focus publication
                # so explicitly calculate the number of citations to the focus publication
                cocite_counts[0,cited2int[focus_pid]] = pub2ref.loc[isin_sorted(pub2ref['CitedPublicationId'].values, np.sort([focus_pid]))]['CitingPublicationId'].nunique()

            # credit share is the matrix product of the credit_allocation_mat with cocite_counts
            credit_share = np.squeeze(np.asarray(credit_allocation_mat.dot(cocite_counts.T)))

            # normalize the credit share vector to sum to 1
            if normed:
                credit_share = credit_share/credit_share.sum(axis=0)

            if temporal:
                return credit_share, author2int, years
            else:
                return credit_share, author2int
        else:
            if temporal:
                years = np.sort(pub2ref.loc[pub2ref['CitedPublicationId'] == focus_pid]['CitingYear'].unique())
                return np.array([[None for y in years] for a in author2int]), author2int, years
            else:
                return np.array([None for a in author2int]), author2int

    elif focus_authors.shape[0] == 1:
        if temporal:
            years = np.sort(pub2ref.loc[pub2ref['CitedPublicationId'] == focus_pid]['CitingYear'].unique())
            return np.ones(shape=(1,years.shape[0])), author2int, years
        else:
            return np.array([1.0]), author2int