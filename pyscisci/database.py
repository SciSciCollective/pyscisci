# -*- coding: utf-8 -*-
"""
.. module:: datasource
    :synopsis: Set of classes to work with different bibliometric data sources

.. moduleauthor:: Alex Gates <ajgates42@gmail.com>
 """
import os
import json
import gzip
from collections import defaultdict

import pandas as pd
import numpy as np
from nameparser import HumanName

from pyscisci.utils import isin_sorted, zip2dict, load_int, load_float, groupby_count
from pyscisci.citationanalysis import *
from pyscisci.datasource.readwrite import load_preprocessed_data, append_to_preprocessed_df


class BibDataBase(object):

    """
    Base class for all bibliometric database interfaces.

    """

    def __init__(self, path2database = '', keep_in_memory = False):

        self.path2database = path2database
        self.keep_in_memory = keep_in_memory

        self._affiliation_df = None
        self._pub_df = None
        self._author_df = None
        self._pub2year = None
        self._pub2ref_df = None
        self._author2pub_df = None
        self._paa_df = None
        self._pub2refnoself_df = None

    @property
    def affiliation_df(self):
        if self._affiliation_df is None:
            if self.keep_in_memory:
                self._affiliation_df = self.load_affiliations()
            else:
                return self.load_affiliations()

        return self._affiliation_df


    @property
    def author_df(self):
        if self._author_df is None:
            if self.keep_in_memory:
                self._author_df = self.load_authors()
            else:
                return self.load_authors()

        return self._author_df

    @property
    def pub_df(self):
        if self._pub_df is None:
            if self.keep_in_memory:
                self._pub_df = self.load_publications()
            else:
                return self.load_publications()

        return self._pub_df

    @property
    def pub2year(self):
        if self._pub2year is None:
            if self.keep_in_memory:
                self._pub2year = self.load_pub2year()
            else:
                return self.load_pub2year()

        return self._pub2year

    @property
    def journal_df(self):
        if self._journal_df is None:
            if self.keep_in_memory:
                self._journal_df = self.load_journals()
            else:
                return self.load_journals()

        return self._journal_df

    @property
    def pub2ref_df(self):
        if self._pub2ref_df is None:
            if self.keep_in_memory:
                self._pub2ref_df = self.load_references()
            else:
                return self.load_references()

        return self._pub2ref_df

    @property
    def pub2refnoself_df(self):
        if self._pub2refnoself_df is None:
            if self.keep_in_memory:
                self._pub2refnoself_df = self.load_references(noselfcite=True)
            else:
                return self.load_references(noselfcite=True)

        return self._pub2refnoself_df

    @property
    def paa_df(self):
        if self._paa_df is None:
            if self.keep_in_memory:
                self._paa_df = self.load_publicationauthoraffiliation()
            else:
                return self.load_publicationauthoraffiliation()

        return self._paa_df

    @property
    def author2pub_df(self):
        if self._paa_df is None:
            if self.keep_in_memory:
                self._paa_df = self.load_publicationauthoraffiliation(columns = ['AuthorId', 'PublicationId'],
                    duplicate_subset = ['AuthorId', 'PublicationId'], dropna = ['AuthorId', 'PublicationId'])
            else:
                return self.load_publicationauthoraffiliation(columns = ['AuthorId', 'PublicationId'],
                    duplicate_subset = ['AuthorId', 'PublicationId'], dropna = ['AuthorId', 'PublicationId'])

        return self._paa_df

    @property
    def pub2field_df(self):
        if self._pub2field_df is None:
            if self.keep_in_memory:
                self._pub2field_df = self.load_pub2field()
            else:
                return self.load_pub2field()

        return self._pub2field_df

    @property
    def fieldinfo_df(self):
        if self._fieldinfo_df is None:
            if self.keep_in_memory:
                self._fieldinfo_df = self.load_fieldinfo()
            else:
                return self.load_fieldinfo()

        return self._fieldinfo_df

    ## Basic Functions for loading data from either preprocessed sources or the raw database files

    def load_affiliations(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'affiliation')):
            return load_preprocessed_data('affiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_affiliations()

    def load_authors(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, process_name = True):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'author')):
            return load_preprocessed_data('author', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_authors(process_name=process_name)

    def load_publications(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'publication')):
            return load_preprocessed_data('publication', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_publications()

    def load_pub2year(self):

        if os.path.exists(os.path.join(self.path2database, 'pub2year.json.gz')):
            with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'r') as infile:
                pub2year = json.loads(infile.read().decode('utf8'))
            return {self.PublicationIdType(k):int(y) for k,y in pub2year.items() if not y is None}

    def load_journals(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'journal')):
            return load_preprocessed_data('journal', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_publications()

    def load_references(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', noselfcite = False, dropna = None):

        if noselfcite:
            fileprefix = 'pub2refnoself'
        else:
            fileprefix = 'pub2ref'

        if preprocess and os.path.exists(os.path.join(self.path2database, fileprefix)):
            return load_preprocessed_data(fileprefix, self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_references()

    def load_publicationauthoraffiliation(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'publicationauthoraffiliation')):
            return load_preprocessed_data('publicationauthoraffiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_publicationauthoraffiliation()

    def load_pub2field(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'pub2field')):
            return load_preprocessed_data('pub2field', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_fields()

    def load_fieldinfo(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
            return load_preprocessed_data('fieldinfo', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_fields()

    """
    To be rewritten for each specific data source (MAG, WOS, etc.)
    """

    def parse_affiliations(self, preprocess = False):
        #TODO: Error Raising when database isnt defined
        pass

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6):
        pass

    def parse_publications(self, preprocess = False, num_file_lines=10**7):
        pass

    def parse_references(self, preprocess = False, num_file_lines=10**7):
        pass

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=10**7):
        pass

    def parse_fields(self, preprocess = False, num_file_lines=10**7):
        pass



    # Analysis

    def author_productivity(self, df=None, colgroupby = 'AuthorId', colcountby = 'PublicationId'):
        if df is None:
            df = self.author2pub_df

        newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['Productivity']*2)
        return groupby_count(df, colgroupby, colcountby, unique=True).rename(columns=newname_dict)

    def author_yearly_productivity(self, df=None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId'):
        if df is None:
            df = self.author2pub_df

        newname_dict = zip2dict([str(colcountby)+'Count', '0'], ['YearlyProductivity']*2)
        return groupby_count(df, [colgroupby, datecol], colcountby, unique=True).rename(columns=newname_dict)

    def author_career_length(self, df = None, colgroupby = 'AuthorId', colrange = 'Year'):
        if df is None:
            df = self.author2pub_df

        newname_dict = zip2dict([str(colrange)+'Range', '0'], ['CareerLength']*2)
        return groupby_range(df, colgroupby, colrange).rename(columns=newname_dict)

    def author_productivity_trajectory(self, df =None, colgroupby = 'AuthorId', datecol = 'Year', colcountby = 'PublicationId'):
        if df is None:
            df = self.author2pub_df

        return compute_yearly_productivity_traj(df, colgroupby = colgroupby)

    def author_hindex(self, df = None, colgroupby = 'AuthorId', colcountby = 'Ctotal'):
        if df is None:
            df = self.author2pub_df.merge(self.impact_df[['AuthorId', colcountby]], on='PublicationId', how='left')
        return compute_hindex(df, colgroupby = colgroupby, colcountby = colcountby)


    def compute_impact(self, preprocess=True, citation_horizons = [5,10], noselfcite = True):

        # first load the publication year information
        pub2year = self.load_pub2year()

        # now get the reference list and merge with year info
        pub2ref = self.pub2ref_df

        # drop all citations that happend before the publication year
        pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) >= pub2year.get(citedpid, 0) for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

        # calcuate the total citations
        citation_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
        citation_df.rename(columns={'CitingPublicationIdCount':'Ctotal', 'CitedPublicationId':'PublicationId'}, inplace=True)

        # go from the larest k down
        for k in np.sort(citation_horizons)[::-1]:

            # drop all citations that happend after the k
            #pub2ref = pub2ref.loc[pub2ref['CitingPublicationYear'] <= pub2ref['CitedPublicationYear'] + k]
            pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) <= pub2year.get(citedpid, 0) + k for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

            # recalculate the impact
            k_citation_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
            k_citation_df.rename(columns={'CitingPublicationIdCount':'C{}'.format(k), 'CitedPublicationId':'PublicationId'}, inplace=True)

            citation_df = citation_df.merge(k_citation_df, how='left', on='PublicationId')

        # get the Cited Year
        citation_df['Year'] = [pub2year.get(pid, 0) for pid in citation_df['PublicationId'].values]


        if noselfcite:
            del pub2ref
            pub2ref = self.pub2refnoself_df

            # drop all citations that happend before the publication year
            pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) >= pub2year.get(citedpid, 0) for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

            # calcuate the total citations
            citation_noself_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
            citation_noself_df.rename(columns={'CitingPublicationIdCount':'Ctotal_noself', 'CitedPublicationId':'PublicationId'}, inplace=True)

            # go from the larest k down
            for k in np.sort(citation_horizons)[::-1]:

                # drop all citations that happend after the k
                #pub2ref = pub2ref.loc[pub2ref['CitingPublicationYear'] <= pub2ref['CitedPublicationYear'] + k]
                pub2ref = pub2ref.loc[[pub2year.get(citingpid, 0) <= pub2year.get(citedpid, 0) + k for citingpid, citedpid in pub2ref[['CitingPublicationId', 'CitedPublicationId']].values]]

                # recalculate the impact
                k_citation_df = groupby_count(pub2ref, colgroupby='CitedPublicationId', colcountby='CitingPublicationId', unique=True )
                k_citation_df.rename(columns={'CitingPublicationIdCount':'C{}_noself'.format(k), 'CitedPublicationId':'PublicationId'}, inplace=True)

                citation_noself_df = citation_noself_df.merge(k_citation_df, how='left', on='PublicationId')

        citation_df = citation_df.merge(citation_noself_df, how='left', on='PublicationId')

        # set all nan to 0
        citation_df.fillna(0, inplace=True)

        if preprocess:

            if not os.path.exists(os.path.join(self.path2database, 'impact')):
                os.mkdir(os.path.join(self.path2database, 'impact'))

            for y, cdf in citation_df.groupby('Year', sort=True):
                cdf.to_hdf(os.path.join(self.path2database, 'impact', 'impact{}.hdf'.format(y)), mode='w', key ='impact')

        else:
            return citation_df




    def compute_teamsize(self, save2pubdf = True):

        pub2teamsize = self.author2pub_df.groupby('PublicationId', sort=False)['AuthorSequence'].max().astype(int).to_frame().reset_index().rename(columns={'AuthorSequence':'TeamSize'})

        if save2pubdf:
            append_to_preprocessed_df(pub2teamsize, self.path2database, 'publication')

        return pub2teamsize

    def remove_selfcitations(self, preprocess = True):

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2refnoself')):
                os.mkdir(os.path.join(self.path2database, 'pub2refnoself'))

        pub2authors = defaultdict(set)
        for pid, aid in self.author2pub_df[['PublicationId', 'AuthorId']].values:
            pub2authors[pid].add(aid)

        fullrefdf = []
        # loop through all pub2ref files
        Nreffiles = sum('pub2ref' in fname for fname in os.listdir(os.path.join(path2mag, 'pub2ref')))
        for ifile in range(Nreffiles):
            refdf = pd.read_hdf(os.path.join(path2mag, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)))

            # get citing cited pairs with no common authors
            noselfcite = np.array([len(pub2authors[citingpid] & pub2authors[citedpid]) == 0 for citingpid, citedpid in refdf.values])

            # keep only citing-cited pairs without a common author
            refdf = refdf.loc[noselfcite]

            if preprocess:
                refdf.to_hdf(os.path.join(path2mag, 'pub2refnoself', 'pub2refnoself{}.hdf'.format(ifile)), key = 'pub2ref')
            else:
                fullrefdf.append(refdf)

        if not preprocess:
            return pd.concat(fullrefdf)

    def compute_yearly_citations(self, preprocess = True, verbose = False):

        if verbose:
            print("Starting Computation of Yearly Citations")

        # first load the publication year information
        pub2year = self.pub2year

        # now get the reference list and merge with year info
        pub2ref = self.pub2ref_df

        pub2ref['CitingYear'] = [pub2year.get(citingpid, 0) for citingpid in pub2ref['CitingPublicationId'].values]

        # drop all citations that happend before the publication year
        pub2ref = pub2ref.loc[[citingyear >= pub2year.get(citedpid, 0) for citingyear, citedpid in pub2ref[['CitingYear', 'CitedPublicationId']].values]]

        if verbose:
            print("Yearly Citation Data Prepared")

        # calcuate the total citations
        citation_df = groupby_count(pub2ref, colgroupby=['CitedPublicationId', 'CitingYear'], colcountby='CitingPublicationId', unique=True )
        citation_df.rename(columns={'CitingPublicationIdCount':'YearlyCitations', 'CitedPublicationId':'PublicationId'}, inplace=True)

        # get the Cited Year
        citation_df['CitedYear'] = [pub2year.get(pid, 0) for pid in citation_df['PublicationId'].values]

        citation_df.sort_values(by=['CitedYear', 'CitedPublicationId', 'CitingYear'], inplace=True)

        if verbose:
            print("Yearly Citations Found")

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'temporalimpact')):
                os.mkdir(os.path.join(self.path2database, 'temporalimpact'))

            for y, cdf in citation_df.groupby('CitedYear', sort=True):
                cdf.to_hdf(os.path.join(self.path2database, 'temporalimpact', 'temporalimpact{}.hdf'.format(y)), mode='w', key ='temporalimpact')

            if verbose:
                print("Yearly Citations Saved")

        else:
            return citation_df

