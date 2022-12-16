Bibliometrics
======================
*pySciSci* facilitates the analysis of publications, authors, citations and as well as citation time-series, fixed time window citation analysis, and citation count normalization by year and field. 


Publications and Citations
---------------------------
The *pySciSci* package facilitates the analysis of interrelationships between publications as captured by references and citations.

For example, the most common measure of scientific impact is the citation count, or the number of times a publication has been referenced by other publications.  Variations also include citation time-series, fixed time window citation analysis, citation count normalization by year and field, and citation ranks.  More advanced methods fit models to citation timeseries, such as in the prediction of the long-term citation counts to a publication :cite:`wang2013longterm`, or in the assignment of the sleeping beauty score :cite:`ke2015sleepingbeauty`.  The package also removes of self-citations occurring between publications by the same author.

More advanced metrics capture the diversity in the citation interrelationships between publications.  These measures include the Rao-Stirling reference interdisciplinary :cite:`stirling2007diversity`, novelty & conventionality :cite:`uzzi2013atypical`, and the disruption index :cite:`funk2017dynamic`, :cite:`wu2019largeteams`.

.. automodule:: pyscisci.methods.publication
   :members:



Author-centric Methods
----------------------

The sociology of science has analyzed scientific careers in terms of individual incentives, productivity, competition, collaboration, and success.  The *pySciSci* package facilitates author career analysis through both aggregate career statistics and temporal career trajectories.  Highlights include the H-index :cite:`hirsch2005index`, Q-factor :cite:`sinatra2016quantifying`, yearly productivity trajectories :cite:`way2017misleading`, collective credit assignment :cite:`shen2014collective`, and hot-hand effect :cite:`liu2018hot`.


.. automodule:: pyscisci.methods.author
   :members:


