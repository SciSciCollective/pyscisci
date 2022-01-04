# -*- coding: utf-8 -*-
"""
.. module:: career topics
    :synopsis: Calculate the co-citing network and detect career communities.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

import sys

import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as spsparse


# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.utils import isin_sorted
from pyscisci.network import cociting_network
from pyscisci.sparsenetworkutils import largest_connected_component_vertices

## Career Cociting Network
def career_cociting_network_topics(paa, pub2ref, randomize=None, return_network=False, show_progress=False):
    """
    This function calculates the topics throughout a career based on the co-citing network (two publications are linked if they share a reference).  
    See :cite:`zeng2019topicswitch` for details.

    Parameters
    ----------
    paa : dataframe
        The publication author affiliation linkages for the focus author.

    pub2ref : dataframe
        The citing-cited publication linkages which contains the citing articles from the focus author.

    randomize : int, default None
        See for random initialization of the community detection algorithm.

    return_network : bool, default False
        Return the networkx object of the co-citing network.

    show_progress : bool, default False
        Show calculation progress. 

    Returns
    ----------
    switching_career : DataFrame
        The paa with topic number included (topics are detected using the Louvain algorithm in the co-citing network).

    cociting_net : networkx.Graph(), optional
        If 'return_network == True' then the cociting network is returned as a networkx graph object.

    """
    try:
        from cdlib import algorithms
    except ImportError:
        raise ImportError("Optional package cdlib needed for this analysis: pip install cdlib")

    try:
        from clusim.clustering import Clustering
    except ImportError:
        raise ImportError("Optional package clusim needed for this analysis: pip install clusim")

    focus_pub_ids = np.sort(paa['PublicationId'].unique())

    # find the co-citing network
    cociting_adjmat, cociting2int = cociting_network(pub2ref, focus_pub_ids=focus_pub_ids, 
        focus_constraint='citing', 
        cited_col_name = 'CitedPublicationId', 
        citing_col_name = 'CitingPublicationId')

    # now take the largest connected component
    lcc_nodes = largest_connected_component_vertices(cociting_adjmat)

    remapnodes = {nid:i for i, nid in enumerate(lcc_nodes)}
    cociting2int = {pid:remapnodes[i] for pid, i in cociting2int.items() if not remapnodes.get(i, None) is None}

    lcc_cociting_adjmat = spsparse.csr_matrix(cociting_adjmat)[lcc_nodes][:,lcc_nodes]

    # remove self-loops and binarize
    lcc_cociting_adjmat.setdiag(0)
    lcc_cociting_adjmat.data[lcc_cociting_adjmat.data >1] = 1
    lcc_cociting_adjmat.eliminate_zeros()

    lcc_cociting_net = nx.Graph(lcc_cociting_adjmat)
    coms = algorithms.louvain(lcc_cociting_net, resolution=1., randomize=randomize)
    louvain_communities = Clustering().from_cluster_list(coms.communities)

    pub2topiccomm = {pid:list(louvain_communities.elm2clu_dict[pid])[0] for pid in lcc_cociting_net.nodes()}

    switching_career = paa[['PublicationId', 'AuthorId', 'Year']].copy()
    switching_career.drop_duplicates(subset=['PublicationId'], inplace=True)
    switching_career = switching_career.loc[isin_sorted(switching_career['PublicationId'].values, np.sort(list(cociting2int.keys())))]
    
    switching_career['TopicCommunity'] = [pub2topiccomm[cociting2int[pid]] for pid in switching_career['PublicationId'].values] 

    switching_career['Degree'] = [lcc_cociting_net.degree()[cociting2int[pid]] for pid in switching_career['PublicationId'].values] 

    switching_career.dropna(inplace=True)

    switching_career.sort_values('Year', inplace=True)

    if return_network:
        nx.set_node_attributes(lcc_cociting_net, pub2topiccomm, "TopicCommunity")
        nx.set_node_attributes(lcc_cociting_net, {i:pid for pid,i in cociting2int.items()}, "PublicationId")

        return switching_career, lcc_cociting_net
    else:
        return switching_career
    