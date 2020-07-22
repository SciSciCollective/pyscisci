# -*- coding: utf-8 -*-
"""
.. module:: embedding
    :synopsis: The main Embedding Class

.. moduleauthor:: Jisung Yoon <jisung.yoon92@gmail.com>
 """
import json
import sys

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse, spatial

# For now, I dont think we need to make the full pySciSci package dependent on these packages
try:
    import numba
except ImportError:
    raise ImportError(
        "Please install numba to take full advantage of fast embedding tools. \n pip install numba"
    )

try:
    from gensim.models import Word2Vec
except ImportError:
    raise ImportError(
        "Please install gensim to take full advantage of fast embedding tools. \n pip install gensim"
    )

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "Please install networkx if you want to take an input as networkx object. \n pip install networkx"
    )


class Node2Vec(object):
    """
    Node2Vec node embedding object
    Paper: https://snap.stanford.edu/node2vec/
    This code is originated from https://github.com/yy/residual-node2vec written by Sadamori Kojaku,
    and modified for its own purpose.
    
    Node2vec works as follows.
    1. Given the network, it generates random walkers on the network.
    2. Random walkers start from each node and travel around the network until the length of the random-walker reaches walk_length. (Total num_walks * walk_length)
    3. Get the embedding of each node with passing these sentences into Gensim Word2vec package (SGNS model).
    """

    def __init__(
        self,
        A,
        entity2int,
        num_walks=10,
        walk_length=80,
        p=1.0,
        q=1.0,
        dimensions=128,
        window_size=10,
        epoch=1,
        workers=1,
    ):
        """
        :param A: Sparse Adjacecny Matrix
        :param entity2int: mapping dict for the network
        :param num_walks: Number of random walker per node
        :param walk_length: Length(number of nodes) of random walker
        :param p: the likelihood of immediately revisiting a node in the walk
        :param q: search to differentiate between “inward” and “outward” nodes in the walk
        :param dimensions: Dimension of embedding vectors
        :param window_size: Maximum distance between the current and predicted node in the network
        :param epoch: Number of epochs over the walks
        :param workers: Number of CPU cores that will be used in training
        """

        self.A = to_csr_adjacency_matrix(
            A
        )  # transform input to csr_matrix for calculation
        self.entity2int = entity2int

        # parameters for random walker
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q

        # parameters for learning embeddings
        self.dimensions = dimensions
        self.window_size = window_size
        self.epoch = epoch

        # computing configuration and path
        self.workers = workers
        self.by_pass_mode = (
            True if p == q == 1 else False
        )  # for the random-walker with p=q=1, which is an unbiased random walker

        self.simulate_walks()

    def simulate_walks(self):
        self.num_nodes = self.A.shape[0]

        if self.by_pass_mode:
            self.walks = simulate_walk(
                self.A,
                num_walk=self.num_walks,
                walk_length=self.walk_length,
                start_node_ids=np.arange(self.num_nodes),
                workers=self.workers,
            )

        else:
            Asupra, node_pairs = construct_line_net_adj(self.A)

            # Find the starting node ids
            start_node_ids = np.where(node_pairs[:, 0] == node_pairs[:, 1])[0]
            # Simulate walks
            trajectories = simulate_walk(
                Asupra,
                num_walk=self.num_walks,
                walk_length=self.walk_length,
                start_node_ids=start_node_ids,
                workers=self.workers,
            )

            # Convert supra-node id to the id of the nodes in the original net
            self.walks = node_pairs[trajectories.reshape(-1), 1].reshape(
                trajectories.shape
            )
        self.walks = self.walks.astype(str).tolist()

    def learn_embedding(self):
        """
        Learning an embedding of nodes in the base graph.
        :return self.embedding: Embedding of nodes in the latent space.
        """
        self.model = Word2Vec(
            self.walks,
            size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            iter=self.epoch,
        )
        self.embedding = {
            entity: self.model.wv[str(index)]
            for entity, index in self.entity2int.items()
        }

        return self.embedding

    def save_embedding(self, file_name):
        """
        :param file_name: name of file_name
        """
        self.model.wv.save_word2vec_format("{}.w2v".format(file_name))
        with open("{}_mapper.json", format(file_name), "w") as fp:
            json.dump(self.entity2int, fp)


# utility function for convert the type
def to_csr_adjacency_matrix(net):
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)


# fast simulate walk
def simulate_walk(
    A, num_walk, walk_length, start_node_ids, workers=1,
):
    """
    Simulate PageRank random walk on the given network.
    The function is compatible with the weighted and directed networks.
    We exploit the sparse.sparse.csr_matrix
    data structure to speed up the simulation of random walkers.
    See `_csr_walk` for details.
    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        Adjacency matrix for the network.
        Self-loops must be removed.
    num_walk: int
        Number of walkers
    walk_length : int
        Number of steps for a single walker
    workers : int
        Number of CPUs
    Return
    ------
    trajectories : numpy.ndarray
        Trajectory of random walker.
        Each row indicates the sequence of nodes that a walker visits.
    """

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csr_matrix")

    # Extract the information on the csr matrix
    A_indptr = A.indptr
    A_indices = A.indices
    A_data = A.data.astype(np.float64)

    # Compute the transition probability priori
    num_nodes = A.shape[0]
    A_data = _calc_cumulative_trans_prob(A_indptr, A_data, num_nodes)

    if start_node_ids is None:
        start_node_ids = np.arange(num_nodes)

    # Parallelize the process
    if workers == 1:
        trajectories = [
            _csr_walk(A_indptr, A_indices, A_data, start_node_ids, walk_length,)
            for i in range(num_walk)
        ]
    else:
        trajectories = Parallel(n_jobs=workers)(
            delayed(_csr_walk)(
                A_indptr, A_indices, A_data, start_node_ids, walk_length,
            )
            for i in range(num_walk)
        )
    return np.vstack(trajectories)


@numba.jit(nopython=True, cache=True)
def _csr_walk(
    A_indptr, A_indices, A_data, start_node_ids, walk_length,
):
    """
    Simulate random walk on the given network.
    We exploit the scipy.sparse.csr_matrix data structure to speed
    up the simulation of random walks. Specifically, consider
    scipy.sparse.csr_matrix, A, representing the adjacency matrix.
    A has the following 3 attributes:
    - A.indices
    - A.data
    - A.indptr
    A.indices and A.data store the column IDs and the value of
    the non-zero element, respectively. A.indptr[i] is the first
    index of the A.indices for the ith row. One can find the neighbors
    of node i by A.indices[A.indptr[i]:A.indptr[i+1]] and the edge weight
    by A.data[A.indptr[i]:A.indptr[i+1]]
    The data structure is memory efficient and we exploit the efficiency
    to speed up the random walk process.
    Furthermore, we use `numba` to further speed up the random walk process.
    Parameters
    ----------
    A_indptr : numpy.ndarray
        A_indptr is given by A.indptr, where A is the scipy.sparse.csr_matrix
        representing the adjacency matrix for the original network
    A_indices : numpy.ndarray
        A_indicesr is given by A.indices, where A is the
        scipy.sparse.csr_matrix as in A_indptr
    A_data : numpy.ndarray
        A_data is given by A.data, where A is the
        scipy.sparse.csr_matrix as in A_indptr
    start_node_ids : numpy.ndarray
        ID of the nodes where random walker starts.
    walk_length : int
        Length of a single walk
    restart_prob : float
        Restart probability
    ergodic_proces : bool 
        If set True, the random walker will already teleport back to the starting node when it hits 
        a dangling node (i.e., node with no outgoing edge). If set False, the random walk ends at dangling nodes.
    Return
    ------
    L : numpy.ndarray
        Size (satrt_node_ids.size, walk_length).
        L[i,t] indicates the node at which the walker 
        starting from start_node_ids[i] visits at time t
    """

    if start_node_ids.size == 0:
        return np.zeros((1, 1), dtype=np.int32)
    # Simulate the random walk
    L = -np.ones((start_node_ids.size, walk_length), dtype=np.int32)
    for sample_id, start_node in enumerate(start_node_ids):

        visit = start_node

        L[sample_id, 0] = visit
        for t in range(1, walk_length):

            outdeg = A_indptr[visit + 1] - A_indptr[visit]

            # If reaches to an absorbing state, finish the walk
            if outdeg == 0:
                break
            else:
                # find a neighbor
                _next_node = np.searchsorted(
                    A_data[A_indptr[visit] : A_indptr[visit + 1]],
                    np.random.rand(),
                    side="right",
                )
                next_node = A_indices[A_indptr[visit] + _next_node]

            # Record the transition
            L[sample_id, t] = next_node

            # Move
            visit = next_node
    return L


@numba.jit(nopython=True)
def _calc_cumulative_trans_prob(A_indptr, A_data, num_nodes):
    """
    Calculate the transition probability for the random walks on
    the given network. The probability is expressed as the
    cumulative probability mass for a fast random
    drawing of a neighboring node.
    Parameters
    ----------
    A_indptr : numpy.ndarray
        A_indptr is given by A.indptr, where A is the scipy.sparse.csr_matrix
        representing the adjacency matrix for the original network
    A_data : numpy.ndarray
        A_data is given by A.data, where A is the scipy.sparse.csr_matrix
        as in A_indptr
    num_nodes : int
        number of nodes in the network
    Return
    ------
    L : numpy.ndarray
        Size (satrt_node_ids.size, walk_length).
        L[i,t] indicates the node at which the walker
        starting from start_node_ids[i] visits at time t
    """
    for i in numba.prange(num_nodes):

        # Compute the out-deg
        outdeg = np.sum(A_data[A_indptr[i] : A_indptr[i + 1]])

        # Compute the transition probability of walkers
        # we compute the cumulative of the transition probabilities, i.e.,
        # A_data[i,j] is the (cumulative) probability of the edge.
        # sum_{j}^ N A[i,j] = 1.
        A_data[A_indptr[i] : A_indptr[i + 1]] = np.cumsum(
            A_data[A_indptr[i] : A_indptr[i + 1]]
        ) / np.maximum(outdeg, 1e-8)

    return A_data


# This function uses the following subroutines:
# - _construct_node2vec_supra_net_edge_pairs
def construct_line_net_adj(A, p=1, q=1, add_source_node=True):
    """
    Construct the supra-adjacent matrix for the weighted network.
    The random walk process in the Node2Vec is the second order Markov process,
    where a random walker at node i remembers the previously visited node j
    and determines the next node k based on i and j.
    We transform the 2nd order Markov process to the 1st order Markov
    process on supra-nodes. Each supra-node represents a pair of
    nodes (j,i) in the original network, a pair of the previouly
    visited node and current node. We place an edge
    from a supra-node (j,i) to another supra-node (i,k) if the random walker
    can transit from node j, i and to k. The weight of edge is given by the
    unnormalized transition probability for the node2vec random walks.
    """

    if A.getformat() != "csr":
        raise TypeError("A should be in the scipy.sparse.csc_matrix")

    # Preprocessing
    # We efficiently simulate the random walk process using the scipy.csr_matrix
    # data structure. The followings are the preprocess for the simulation.
    A.sort_indices()  # the indices have to be sorted
    A_indptr_csr = A.indptr.astype(np.int32)
    A_indices_csr = A.indices.astype(np.int32)
    A_data_csr = A.data.astype(np.float32)
    A = A.T
    A.sort_indices()  # the indices have to be sorted
    A_indptr_csc = A.indptr.astype(np.int32)
    A_indices_csc = A.indices.astype(np.int32)
    A_data_csc = A.data.astype(np.float32)
    num_nodes = A.shape[0]

    # Make the edge list for the supra-networks
    supra_edge_list, edge_weight = construct_node2vec_supra_net_edge_pairs(
        A_indptr_csr,
        A_indices_csr,
        A_data_csr,
        A_indptr_csc,
        A_indices_csc,
        A_data_csc,
        num_nodes,
        p=p,
        q=q,
        add_source_node=add_source_node,
    )

    # Remove isolated nodes from the supra-network and
    # re-index the ids.
    supra_nodes, supra_edge_list = np.unique(supra_edge_list, return_inverse=True)
    supra_edge_list = supra_edge_list.reshape((edge_weight.size, 2))

    # Each row indicates the node pairs in the original net that constitute the supra node
    src_trg_pairs = (np.vstack(divmod(supra_nodes, num_nodes)).T).astype(int)

    # Construct the supra-adjacency matrix
    supra_node_num = supra_nodes.size
    Aspra = sparse.csr_matrix(
        (edge_weight, (supra_edge_list[:, 0], supra_edge_list[:, 1])),
        shape=(supra_node_num, supra_node_num),
    )
    return Aspra, src_trg_pairs


def sem_axis(emb, positive_entities, negative_entities):
    """
    SemAxis Code
    Paper: https://arxiv.org/abs/1806.05521
    SemAxis is a technique that leverages the latent semantic characteristics of word embeddings to represent 
    The position of terms along a conceptual axis, reﬂecting the relationship of these terms to the concept.
    
    Semaxis works as follows.
    1. Given the set of positive instances, S+, and set of negative instances S-, calculate the average vector of each set (V+, V-).
    2. calculate the semantic axis can be calculated as V_{axis} = V+ - V-, and project another instance into this semantic axis with measuring cosine similarity to the semantic axis.
    
    Higher scores mean that organization w is more closely aligned to S+ than S− .
    """
    mean_positivie_vec = np.array([emb[entity] for entity in positive_entities])
    mean_negative_vec = np.array([emb[entity] for entity in negative_entities])

    target_axis = mean_positivie_vec - mean_negative_vec
    sem_axis_result_dict = {
        k: 1 - spatial.distance.cosine(target_axis, v) for k, v in emb.items()
    }

    return sem_axis_result_dict
