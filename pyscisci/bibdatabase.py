# -*- coding: utf-8 -*-
"""
.. module:: bibdatabase
    :synopsis: The main Bibliometric Database class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import json

from .publication import Publication
from .author import Author
from .journal import Journal
from .affiliation import Affiliation
from .dataset_interface.mag_interface import *


class BibDatabase(object):
    """
    Base class for bibliometric data.

    """

    def __init__(self):

        self._pubdict = dict([])

        self._authordict = dict([])

        self._journaldict = dict([])

        self._affiliationdict = dict([])

        self.reset_properties()


    def reset_properties(self):

        self._start_year = None
        self._end_year = None
        self._publication_years = None


    def update_database(self):
        self.reset_properties()

        self.start_year()
        self.end_year()


    def get_pub(self, pubid):
        return self._pubdict.get(pubid, Publication(self))

    def add_pub(self, pubobject):
        self._pubdict[pubobject.id] = pubobject

    def load_publications(self, fname):
        with open(fname, 'r') as pubfile:
            for line in pubfile:
                self.add_pub(Publication(self).from_dict(json.loads(line)))

    def get_author(self, authorid):
        return self._authordict.get(authorid, Author(self))

    def add_author(self, authorobject):
        self._authordict[authorobject.id] = authorobject

    def load_authors(self, fname):
        with open(fname, 'r') as authorfile:
            for line in authorfile:
                self.add_author(Author(self).from_dict(json.loads(line)))

    def get_journal(self, journalid):
        return self._journaldict.get(journalid, Journal(self))

    def add_jouranl(self, journalobject):
        self._journaldict[journalobject.id] = journalobject

    def get_affiliation(self, affiliationid):
        return self._journaldict.get(affiliationid, Affiliation(self))

    def add_jouranl(self, journalobject):
        self._journaldict[journalobject.id] = journalobject

    @property
    def n_publications(self):
        return len(self._pubdict)

    @property
    def n_authors(self):
        return len(self._authordict)

    @property
    def n_affiliations(self):
        return len(self._affiliationdict)

    @property
    def publication_ids(self):
        for pid in self._pubdict.keys():
            yield pid

    @property
    def publication_years(self):
        if self._publication_years is None:
            self._publication_years = sorted(set(pub.year for pub in self._pubdict.values()))

        return self._publication_years

    @property
    def start_year(self):
        if self._start_year is None:
            self._start_year = min(self.publication_years)
        return self._start_year

    @property
    def end_year(self):
        if self._end_year is None:
            self._end_year = max(self.publication_years)
        return self._end_year


    def publication_generator(self):
        for pubobject in self._pubdict.values():
            return pubobject

    def author_generator(self):
        for author in self._authordict.values():
            return author


