# -*- coding: utf-8 -*-
"""
.. module:: hindex
    :synopsis: The h-index.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

def compute_hindex(database, publications):
    """
    This function calculates the h-index for an author (or more generally, a list of publications).

    The h-index is the number of publications n such that each has been cited at least n times (Hirsch [h]_).

    :param BibDataBase database:
        The bibliometric database object.

    :param list publications:
        The publication list.

    :returns:
        The h-index

    See the Author() object for usage

    References
    ----------
    .. [h] Hirsch, J. E. (2005): "An index to quantify
           an individual's scientific research output",
           *PNAS* 102(46).
           DOI: 10.1073/pnas.0507655102
    """

    citation_counts = sorted(list(database.get_pub(pub).total_citations for pub in publications), reverse = True)
    idx = range(1, len(citation_counts) + 1)

    return sum(p <= c for (c, p) in zip(citation_counts, idx))