# -*- coding: utf-8 -*-
"""
.. module:: cnorm
    :synopsis: The yearly normalized citation count.

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
from collections import defaultdict
from statistics import mean

def compute_yearly_cnorm(database, citations):
    """
    This function calculates the yearly cnorm for a publication.

    The cnorm is the number of citations a publication recieved in a year, divided by the average number of citations
    all publications in the co-citation neighborhood recieved that year (Ke [h]_).

    :param BibDataBase database:
        The bibliometric database object.

    :param list citations:
        The list of citing publications.

    :returns:
        The yearly cnorm

    See the Publication() object for usage

    References
    ----------
    .. [h] Ke, Q., Gates, A.J., Barabasi, A.L. (2020): "title",
           *in submission*.
           DOI: xxx
    """
    citation_pubs = [database.get_pub(pub) for pub in citations]

    # aggregate citations by year
    year2citations = defaultdict(list)
    for cite in citation_pubs:
        year2citations[cite.year].append(cite)

    yearly_cnorm = defaultdict(float)
    for year, yearly_citation_list in year2citations.items():
        yearly_cocitation = set(ref for pub in yearly_citation_list for ref in pub.references)
        yearly_cnorm[year] = float(len(year2citations[year])) / mean(database.get_pub(ref).yearly_citations.get(year, 0) for ref in yearly_cocitation)

    return yearly_cnorm

