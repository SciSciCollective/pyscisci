# -*- coding: utf-8 -*-
"""
.. module:: publication
    :synopsis: The main Publication class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

from collections import Counter

from .metrics.disruption import compute_disruption
from .metrics.cnorm import compute_yearly_cnorm

class Publication(object):
    """
    Base class for publications.

    :param dict elm2clu_dict: optional
        Initialize based on an elm2clu_dict: { elementid: [clu1, clu2, ... ] }.
        The value is a list of clusters to which the element belongs.

    """

    def __init__(self, database, pubdict = {}):

        self.database = database

        self.from_dict(pubdict)

        self.reset_properties()

    def reset_properties(self, pubdict = {}):

        self._total_citations = pubdict.get('_total_citations', None)
        self._yearly_citations = pubdict.get('_total_citations', None)
        self._ck = pubdict.get('_ck', {})
        self._disruption = pubdict.get('_disruption', None)
        self._cocited = pubdict.get('_cocited', None)
        self._yearly_cnorm = pubdict.get('_yearly_cnorm', None)

    def from_dict(self, pubdict):

        self.id = pubdict.get('id', 0)
        self.year = pubdict.get('year', None)
        self.sort_date = pubdict.get('sort_date', None)
        self.title = pubdict.get('title', None)
        self.journal = pubdict.get('journal', None)
        self.authors = pubdict.get('authors', [])
        self.references = pubdict.get('references', [])
        self.citations = pubdict.get('citations', [])
        self.doctype = pubdict.get('doctype', [])
        self.volume = pubdict.get('volume', None)
        self.issue = pubdict.get('issue', None)
        self.pages = pubdict.get('pages', None)
        self.abstract = pubdict.get('abstract', None)

        self.metadata = pubdict.get('metadata', {})

        self.reset_properties(pubdict)

        return self

    @property
    def team_size(self):
        return len(self.authors)

    @property
    def total_citations(self):
        if self._total_citations is None:
            self._total_citations = len(self.citations)
        return self._total_citations

    @property
    def yearly_citations(self):
        if self._yearly_citations is None:
            self._yearly_citations = Counter(self.database.get_pub(pub).year for pub in self.citations)
        return self._yearly_citations


    def ck(self, k = 10):
        if self._ck.get(k, None) is None:
            self._ck[k] = sum(self.yearly_citations.get(y,0) for y in range(self.year, self.year + k + 1))
        return self._ck[k]

    @property
    def c10(self):
        return self.ck(10)

    @property
    def c5(self):
        return self.ck(5)

    @property
    def disruption_index(self):
        if self._disruption is None:
            self._disruption = compute_disruption(self.database, self.year, self.references, self.citations)

        return self._disruption

    @property
    def cocited_publications(self):
        if self._cocited is None:
            if len(self.citations) == 0:
                self._cocited = set([])
            else:
                self._cocited = set(ref for cite in self.citations for ref in self.database.get_pub(cite).references)
                if self.id in self._cocited:
                    self._cocited.remove(self.id)
        return self._cocited

    @property
    def yearly_cnorm(self):
        if self._yearly_cnorm is None:
            self._yearly_cnorm = compute_yearly_cnorm(self.database, self.citations)
        return self._yearly_cnorm

    @property
    def cnorm(self, k = 10):
        return sum(self.yearly_cnorm.get(y, 0.0) for y in range(self.year, self.year + k + 1))

    def remove_selfcitations(self):
        author_publications = set(pub for auth in self.authors for pub in self.database.get_author(auth).publications)
        self.citations = list(set(self.citations) - author_publications)

    def __repr__(self):
        return "Publication()"

    def __str__(self):
        return "{}".format(self.id)



