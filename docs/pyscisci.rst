
Welcome to PySciSci's documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :glob:


Installation
===================
This package is available in PyPI. Just run the following command on terminal to install.

>>> pip install pyscisci

You can also source the code directly from the github [project page](https://github.com/ajgates42/pyscisci).


Usage
===================

A first comparison
--------------------------
We start by importing the required modules

>>> from clusim.clustering import Clustering, print_clustering
>>> import clusim.sim as sim

The simplest way to make a Clustering is to use an elm2clu_dict which maps each element.

>>> c1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
>>> c2 = Clustering(elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[1], 4:[2], 5:[2]})

>>> print_clustering(c1)
>>> 01|23|45
>>> print_clustering(c2)
>>> 0|123|45

Finally, the similarity of the two Clusterings can be found using the Jaccard Index.

>>> sim.jaccard_index(c1, c2)
>>> 0.4

Basics of element-centric similarity
--------------------------------------
>>> from clusim.clustering import Clustering, print_clustering
>>> import clusim.sim as sim

>>> c1 = Clustering(elm2clu_dict = {0:[0], 1:[0], 2:[1], 3:[1], 4:[2], 5:[2]})
>>> c2 = Clustering(elm2clu_dict = {0:[0], 1:[1], 2:[1], 3:[1], 4:[1], 5:[2]})

The basic element-centric similarity score with a fixed alpha:

>>> sim.element_sim(c1, c2, alpha = 0.9)
>>> 0.6944444444444443

We can also get the element scores.  Note that since non-numberic elements are allowed, the element scores also returns a dict which maps the elements to the index in the elementScore array.

>>> elementScores, relabeled_elements = sim.element_sim_elscore(c1, c2, alpha = 0.9)
>>> print(elementScores)
>>> [0.5        0.33333333 0.66666667 0.66666667 1.         1.        ]

The above element-centric similarity scores can be (roughly) interpreted as follows:
1. cluster 2 has the same memberships between the clusterings, so elements 4 and 5 have an element-centric similarity of 1.0
2. cluster 0 has one element difference between the clusterings (element 1 moved from cluster 0 to cluster 1), so element 0 has an element-centric similarity of 1/2
3. cluster 1 has one element difference between the clusterings (element 1 moved from cluster 0 to cluster 1), so elements 2 and 3 have an element-centric similarity of 2/3
4. element 1 moved from cluster 0 to cluster 1 so it has an element-centric similarity of 1/3


Additional CluSim examples
--------------------------------------
Many more examples can be found in the jupyter notebooks included with the package:
1. `Using Similarity Measures <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20apply%20all%20similarity%20measures.ipynb>`_
2. `Adjusting The Rand Index for different random models <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20expected%20Rand%20Index%20for%20different%20Random%20Models.ipynb>`_
3. `Adjusting Normalized Mutual Information <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20NMI%20adjustment%20and%20normalization.ipynb>`_
4. `Basics of Elementcentric Similarity <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20basic%20element-centric%20similarity.ipynb>`_
5. `Valid Clusterings <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20valid%20Clustering.ipynb>`_
6. `Application with SciKit Learn <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20application%20with%20scikit-learn.ipynb>`_
7. `Application with Hierarchical Clustering <https://github.com/Hoosier-Clusters/clusim/blob/master/examples/CluSim%20Examples%20-%20application%20with%20hierarchical%20clustering.ipynb>`_


The “Clustering”
=================
.. autoclass:: clusim.clustering.Clustering
    :members:

.. autoclass:: clusim.clusteringerror
    :members:

.. automodule:: clusim.clustering
    :members:


Clustering Generation
======================
.. automodule:: clusim.clugen
   :members:


Clustering Similarity
======================
The different clustering similarity measures available.


Pairwise Counting Measures
--------------------------
.. autofunction:: clusim.sim.contingency_table

.. autofunction:: clusim.sim.count_pairwise_cooccurence
.. autofunction:: clusim.sim.jaccard_index
.. autofunction:: clusim.sim.rand_index
.. autofunction:: clusim.sim.fowlkes_mallows_index
.. autofunction:: clusim.sim.fmeasure
.. autofunction:: clusim.sim.purity_index
.. autofunction:: clusim.sim.classification_error
.. autofunction:: clusim.sim.czekanowski_index
.. autofunction:: clusim.sim.dice_index
.. autofunction:: clusim.sim.sorensen_index
.. autofunction:: clusim.sim.rogers_tanimoto_index
.. autofunction:: clusim.sim.southwood_index
.. autofunction:: clusim.sim.pearson_correlation


Information Theoretic Measures
------------------------------
.. autofunction:: clusim.sim.mi
.. autofunction:: clusim.sim.nmi
.. autofunction:: clusim.sim.vi
.. autofunction:: clusim.sim.rmi

Correction for Chance
------------------------------
.. autofunction:: clusim.sim.corrected_chance
.. autofunction:: clusim.sim.sample_expected_sim
.. autofunction:: clusim.sim.expected_rand_index
.. autofunction:: clusim.sim.adjrand_index
.. autofunction:: clusim.sim.adj_mi
.. autofunction:: clusim.sim.expected_mi


Overlapping Clustering Similarity
----------------------------------
.. autofunction:: clusim.sim.onmi
.. autofunction:: clusim.sim.omega_index
.. autofunction:: clusim.sim.geometric_accuracy
.. autofunction:: clusim.sim.overlap_quality


Element-centric Clustering Similarity
======================================
.. automodule:: clusim.clusimelement
   :members:


References
===========
.. bibliography:: clusimref.bib
