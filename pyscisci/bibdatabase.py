# -*- coding: utf-8 -*-
"""
.. module:: bibdatabase
    :synopsis: The main Bibliometric Database class

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import os
import json
import gzip

from .publication import Publication
from .author import Author
from .journal import Journal
from .affiliation import Affiliation
from dataset_interface.mag_interface import *


class BibDatabase(object):
    """
    Base class for bibliometric data.

    """

    def __init__(self, name='Bibdatabase'):

        self.name = name

        self._pubdict = dict([])

        self._authordict = dict([])

        self._journaldict = dict([])

        self._affiliationdict = dict([])

        self.reset_properties()


    def reset_properties(self):

        self._start_year = None
        self._end_year = None
        self._publication_years = None


    def get_pub(self, pubid):
        return self._pubdict.get(pubid, Publication(self))

    def add_pub(self, pubobject):
        self._pubdict[pubobject.id] = pubobject

    def get_author(self, authorid):
        return self._authordict.get(authorid, Author(self))

    def add_author(self, authorobject):
        self._authordict[authorobject.id] = authorobject

    def get_journal(self, journalid):
        return self._journaldict.get(journalid, Journal(self))

    def add_jouranl(self, journalobject):
        self._journaldict[journalobject.id] = journalobject

    def get_affiliation(self, affiliationid):
        return self._affiliationdict.get(affiliationid, Affiliation(self))

    def add_affiliation(self, affiliationobject):
        self._affiliationdict[affiliationobject.id] = affiliationobject

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

    def write_database_archive(self, path2archive = '', publication_subset = [], author_subset = []):
        if len(author_subset) > 0:
            publication_subset = set(publication_subset).union([pid for aid in author_subset for pid in self._authordict[aid].publications])

        self.archive_authors(path2archive, author_subset)
        self.archive_publications(path2archive, publication_subset)
        self.archive_affiliations(path2archive)
        self.archive_journals(path2archive)

    def archive_authors(self, path2archive = '', author_subset = []):
        # save authors
        author_path = os.path.join(path2archive, self.name+'_authors.jsonl.gz')
        with gzip.open(author_path, 'w') as output:
            if len(author_subset) == 0:
                for author in self.author_generator():
                    output.write(json.dumps(author.to_dict()).encode('utf8') + b"\n")
            else:
                for authorid in author_subset:
                    output.write(json.dumps(self._authordict[authorid].to_dict()).encode('utf8') + b"\n")

    def archive_publications(self, path2archive = '', publication_subset = []):
        # save publications
        pub_path = os.path.join(path2archive, self.name+'_publications.jsonl.gz')
        with gzip.open(pub_path, 'w') as output:
            if len(publication_subset) == 0:
                for pub in self.publication_generator():
                    output.write(json.dumps(pub.to_dict()).encode('utf8') + b"\n")
            else:
                for pubid in publication_subset:
                    output.write(json.dumps(self._pubdict[pubid].to_dict()).encode('utf8') + b"\n")

    def archive_affiliations(self, path2archive = ''):
        # save affiliations
        affil_path = os.path.join(path2archive, self.name+'_affiliations.jsonl.gz')
        with gzip.open(affil_path, 'w') as output:
            for affiliation in self._affiliationdict.values():
                output.write(json.dumps(affiliation.to_dict()).encode('utf8') + b"\n")

    def archive_journals(self, path2archive = ''):
        # save affiliations
        journal_path = os.path.join(path2archive, self.name+'_journals.jsonl.gz')
        with gzip.open(journal_path, 'w') as output:
            for journal in self._journaldict.values():
                output.write(json.dumps(journal.to_dict()).encode('utf8') + b"\n")

    def load_database_archive(self, path2archive=''):
        self.load_affiliations(path2archive)
        self.load_authors(path2archive)
        self.load_publications(path2archive)
        self.load_journals(path2archive)

    def load_publications(self, fname = None, path2archive = ''):

        if fname is None:
            fname = self.name+'_publications.jsonl.gz'

        pub_path = os.path.join(path2archive, fname)
        if os.path.splitext(fname)[-1] == '.gz':
            with gzip.open(pub_path, 'rb') as infile:
                for line in infile:
                    self.add_pub(Publication(self, json.loads(line)))

        else:
            with open(pub_path, 'r') as infile:
                for line in infile:
                    self.add_pub(Publication(self, json.loads(line)))

    def load_authors(self, fname = None, path2archive = ''):

        if fname is None:
            fname = self.name+'_authors.jsonl.gz'

        author_path = os.path.join(path2archive, fname)
        if os.path.splitext(fname)[-1] == '.gz':
            with gzip.open(author_path, 'rb') as infile:
                for line in infile:
                    self.add_author(Author(self, json.loads(line)))

        else:
            with open(author_path, 'r') as infile:
                for line in infile:
                    self.add_author(Author(self, json.loads(line)))

    def load_affiliations(self, fname = None, path2archive = ''):

        if fname is None:
            fname = self.name+'_affiliations.jsonl.gz'

        affil_path = os.path.join(path2archive, fname)
        if os.path.splitext(fname)[-1] == '.gz':
            with gzip.open(affil_path, 'rb') as infile:
                for line in infile:
                    self.add_affiliation(Affiliation(self, json.loads(line)))

        else:
            with open(affil_path, 'r') as infile:
                for line in infile:
                    self.add_affiliation(Author(self, json.loads(line)))

    def load_journals(self, fname = None, path2archive = ''):

        if fname is None:
            fname = self.name+'_journals.jsonl.gz'

        journal_path = os.path.join(path2archive, fname)
        if os.path.splitext(fname)[-1] == '.gz':
            with gzip.open(journal_path, 'rb') as infile:
                for line in infile:
                    self.add_journal(Journal(self, json.loads(line)))

        else:
            with open(journal_path, 'r') as infile:
                for line in infile:
                    self.add_journal(Journal(self, json.loads(line)))

