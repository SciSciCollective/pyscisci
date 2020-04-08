# -*- coding: utf-8 -*-
"""
.. module:: author
    :synopsis: The main Author class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """

from collections import Counter, defaultdict

from .publicationcollection import PublicationCollection

from .metrics.hindex import compute_hindex


class Author(PublicationCollection):
    """
    Base class for authors.

    :param dict elm2clu_dict: optional
        Initialize based on an elm2clu_dict: { elementid: [clu1, clu2, ... ] }.
        The value is a list of clusters to which the element belongs.

    """

    def __init__(self, database):

        self.database = database

        self.from_dict({})


    def reset_properties(self, author_dict = {}):

        self.reset_collection_properties(author_dict)

        self._author_timeline = author_dict.get('_author_timeline', None)
        self._hindex = author_dict.get('_hindex', None)
        self._coauthorship = author_dict.get('_coauthorship', None)

    def from_dict(self, author_dict):

        self.id = author_dict.get('id', 0)
        self.fullname = author_dict.get('fullname', '')
        self.lastname = author_dict.get('lastname', '')
        self.firstname = author_dict.get('firstname', '')
        self.middlename = author_dict.get('middlename', '')

        self.metadata = author_dict.get('metadata', None)

        self.publications = author_dict.get('publications', [])
        self.affiliation2pub = author_dict.get('affiliation2pub', defaultdict(set))

        self.reset_properties(author_dict)

        return self

    def add_publication(self, pubid):
        if not pubid in self.publications:
            self.publications.append(pubid)

    @property
    def author_timeline(self):
        if self._author_timeline is None:
            self._author_timeline = []
            for pub in self.publications:
                pubobj = self.database.get_pub(pub)
                self._author_timeline.append([pubobj.year, pubobj.c10])
        return self._author_timeline


    @property
    def hindex(self):
        if self._hindex is None:
            self._hindex = compute_hindex(self.database, self.publications)
        return self._hindex

    @property
    def coauthor_weights(self):
        if self._coauthorship is None:
            self._coauthorship = Counter(a for p in self.publications for a in p.authors)
            if not self._coauthorship.get(self.id, None) is None:
                del self._coauthorship[self.id]
        return self._coauthorship


