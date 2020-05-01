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
from pyscisci.analysis import groupby_count

def load_preprocessed_data(dataname, path2database, columns = None, isindict = None, duplicate_subset = None,
    duplicate_keep = 'last', dropna = None, keep_source_file = False, func2apply = None):

    #TODO: progress bar

    path2files = os.path.join(path2database, dataname)
    if not os.path.exists(path2files):
        # TODO: make a real warning
        print("First preprocess the raw data.")
        return []

    if isinstance(columns, str):
        columns = [columns]

    if isinstance(dropna, str):
        dropna = [dropna]

    if isinstance(duplicate_subset, str):
        duplicate_subset = [duplicate_subset]

    if isinstance(isindict, dict):
        isindict = {isinkey:np.sort(isinlist) for isinkey, isinlist in isindict.items()}


    Nfiles = sum(dataname in fname for fname in os.listdir(path2files))

    data_df = []
    for ifile in range(Nfiles):
        fname = os.path.join(path2files, dataname+"{}.hdf".format(ifile))
        subdf = pd.read_hdf(fname, mode = 'r')

        if isinstance(columns, list):
            subdf = subdf[columns]

        if isinstance(dropna, list):
            subdf.dropna(subset = dropna, inplace = True, how = 'any')

        if isinstance(isindict, dict):
            for isinkey, isinlist in isindict.items():
                subdf = subdf[isin_sorted(subdf[isinkey], isinlist)]

        if isinstance(duplicate_subset, list):
            subdf.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

        if keep_source_file:
            subdf['filetag'] = ifile

        if callable(func2apply):
            func2apply(subdf)

        data_df.append(subdf)

    data_df = pd.concat(data_df)

    if isinstance(duplicate_subset, list):
        data_df.drop_duplicates(subset = duplicate_subset, keep = duplicate_keep, inplace = True)

    return data_df

def append_to_preprocessed_df(newdf, path2database, preprocessname):

    path2files = os.path.join(path2data, preprocessname)

    Nfiles = sum(preprocessname in fname for fname in os.listdir(path2files))

    for ifile in range(Nfiles):
        datadf = pd.read_hdf(os.path.join(path2files, preprocessname + '{}.hdf'.format(ifile)))
        datadf = datadf.merge(newdf, how = 'left')
        datadf.to_hdf(os.path.join(path2files, preprocessname + '{}.hdf'.format(ifile)), key = preprocessname, mode = 'w')



class BibDataSource(object):

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

    @property
    def pub_df(self):
        #TODO: Error Raising when database isnt defined
        return False

    @property
    def pub2ref_df(self):
        #TODO: Error Raising when database isnt defined
        return False

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


class MAG(BibDataSource):
    """
    Base class for Microsoft Academic Graph interface.

    The MAG comes structured into three folders: mag, advanced, nlp.
    Explain downloading etc.

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

    def preprocess(self, dflist = None):
        if dflist is None:
            dflist = ['affiliation', 'author', 'publication', 'reference', 'publicationauthoraffiliation', 'fields']

        if 'affiliation' in dflist:
            self.parse_affiliations(preprocess = True)

        if 'author' in dflist:
            self.parse_authors(preprocess = True)

        if 'publication' in dflist:
            self.parse_publications(preprocess = True)

        if 'reference' in dflist:
            self.parse_references(preprocess = True)

        if 'publicationauthoraffiliation' in dflist:
            self.parse_publicationauthoraffiliation(preprocess = True)

        if 'fields' in dflist:
            self.parse_fields(preprocess=True)

    @property
    def affiliation_df(self):
        if self._affiliation_df is None:
            if self.keep_in_memory:
                self._affiliation_df = self.load_affiliations()
            else:
                return self.load_affiliations()

        return self._affiliation_df


    def load_affiliations(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'affiliation')):
            return load_preprocessed_data('affiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_affiliations()

    def parse_affiliations(self, preprocess = False):

        affil_int_columns = [0, 7, 8]
        affil_str_columns = [3, 4, 5, 6]
        affil_float_columns = [9, 10]

        affil_column_names = ['AffiliationId', 'NumberPublications', 'NumberCitations', 'FullName', 'GridId', 'OfficialPage', 'WikiPage', 'Latitude', 'Longitude']

        affiliation_info = []
        with open(os.path.join(self.path2database, 'mag', 'Affiliations.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                affline = [load_int(sline[i]) for i in affil_int_columns]
                affline += [sline[i] for i in affil_str_columns]
                affline += [load_float(sline[i]) for i in affil_float_columns]
                affiliation_info.append(affline)

        aff_df = pd.DataFrame(affiliation_info, columns = affil_column_names)

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'affiliation')):
                os.mkdir(os.path.join(self.path2database, 'affiliation'))
            aff_df.to_hdf(os.path.join(self.path2database, 'affiliation', 'affiliation0.hdf'), key = 'affiliation', mode = 'w')

        return aff_df

    @property
    def author_df(self):
        if self._author_df is None:
            if self.keep_in_memory:
                self._author_df = self.load_authors()
            else:
                return self.load_authors()

        return self._author_df

    def load_authors(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, process_name = True):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'author')):
            return load_preprocessed_data('author', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_authors(process_name=process_name)

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6):

        author_int_columns = [0, 4, 5, 6]

        author_column_names = ['AuthorId', 'LastKnownAffiliationId', 'NumberPublications', 'NumberCitations', 'FullName']
        if process_name:
            author_column_names += ['LastName', 'FirstName', 'MiddleName']


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'author')):
                os.mkdir(os.path.join(self.path2database, 'author'))

        iauthor = 0
        ifile = 0
        authorinfo = []
        with open(os.path.join(self.path2database, 'mag', 'Authors.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                adata = [load_int(sline[ip]) for ip in author_int_columns] + [sline[2]]
                if process_name:
                    hname = HumanName(unicodedata.normalize('NFD', sline[2]))
                    adata += [hname.last, hname.first, hname.middle]
                authorinfo.append(adata)
                iauthor += 1

                if preprocess and iauthor % num_file_lines == 0:
                    pd.DataFrame(authorinfo, columns = author_column_names).to_hdf(
                        os.path.join(self.path2database, 'author', 'author{}.hdf'.format(ifile)),
                                                                                key = 'author', mode = 'w')

                    ifile += 1
                    authorinfo = []

            author_df = pd.DataFrame(authorinfo, columns = author_column_names)
            if preprocess:
                author_df.to_hdf(os.path.join(self.path2database, 'author', 'author{}.hdf'.format(ifile)),
                                                                            key = 'author', mode = 'w')

        return author_df

    @property
    def pub_df(self):
        if self._pub_df is None:
            if self.keep_in_memory:
                self._pub_df = self.load_publications()
            else:
                return self.load_publications()

        return self._pub_df

    def load_publications(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'publication')):
            return load_preprocessed_data('publication', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_publications()

    @property
    def journal_df(self):
        if self._journal_df is None:
            if self.keep_in_memory:
                self._journal_df = self.load_journals()
            else:
                return self.load_journals()

        return self._journal_df

    def load_journals(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'journal')):
            return load_preprocessed_data('journal', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_publications()

    def load_pub2year(self):

        with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'r') as infile:
            pub2year = json.loads(infile.read().decode('utf8'))
        return {int(k):int(v) for k,v in pub2year.items() if not v is None}

    def parse_publications(self, preprocess = False, num_file_lines=10**7):

        # first do the journal information
        journal_str_col = [2, 4, 5, 6]
        journal_column_names = ['JournalId', 'FullName', 'Issn', 'Publisher', 'Webpage']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'journal')):
                os.mkdir(os.path.join(self.path2database, 'journal'))

        journal_info = []
        with open(os.path.join(path2mag, 'RawTXT/mag', 'Journals.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                jline = [load_int(sline[0])] + [sline[i] for i in journal_str_col]
                journal_info.append(jline)

        journal_df = pd.DataFrame(journal_info, columns = journal_column_names)
        if preprocess:
            journal_df.to_hdf(os.path.join(path2mag, 'journal', 'journal.hdf'), key = 'journal', mode = 'w')

        #now lets do the publication information

        doctype = {'Journal': 'j', 'Book':'b', '':'', 'BookChapter':'bc', 'Conference':'c', 'Dataset':'d', 'Patent':'p', 'Repository':'r'}

        pub_int_columns = [0, 7, 10, 21]
        pub_str_columns = [2, 4, 8, 13, 14]
        pub_column_names = ['PublicationId', 'Year', 'JournalId', 'FamilyId',  'Doi', 'Title', 'Date', 'Volume', 'Issue', 'DocType']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'publication')):
                os.mkdir(os.path.join(self.path2database, 'publication'))

        ipub = 0
        ifile = 0
        pubinfo = []

        pub2year = {}

        with open(os.path.join(self.path2database, 'mag', 'Papers.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                pline = [load_int(sline[ip]) for ip in pub_int_columns] + [sline[ip] for ip in pub_str_columns] + [doctype[sline[3]]]
                pub2year[pline[0]] = pline[1]
                pubinfo.append(pline)
                ipub += 1

                if preprocess and ipub % num_file_lines == 0:
                        pd.DataFrame(pubinfo, columns = pub_column_names).to_hdf(
                            os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)),
                                                                                    key = 'publication', mode = 'w')

                        ifile += 1
                        pubinfo = []

            pub_df = pd.DataFrame(pubinfo, columns = pub_column_names)
            if preprocess:
                pub_df.to_hdf(os.path.join(self.path2database, 'publication', 'publications{}.hdf'.format(ifile)),
                                                                                key = 'publication', mode = 'w')

                with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
                    outfile.write(json.dumps(self.pub2year).encode('utf8'))

        return pub_df

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

    def parse_references(self, preprocess = False, num_file_lines=10**7):

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2ref')):
                os.mkdir(os.path.join(self.path2database, 'pub2ref'))

        iref = 0
        ifile = 0
        pub2ref_info = []
        with open(os.path.join(self.path2database, 'mag', 'PaperReferences.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                pub2ref_info.append([load_int(sline[ip]) for ip in range(2)])
                iref += 1

                if preprocess and iref % num_file_lines == 0:
                    pd.DataFrame(pub2ref_info, columns = ['CitingPublicationId', 'CitedPublicationId']).to_hdf(
                        os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)),
                                                                                key = 'pub2ref', mode = 'w')

                    ifile += 1
                    pub2ref_info = []

            pub2ref_df = pd.DataFrame(pub2ref_info, columns = ['CitingPublicationId', 'CitedPublicationId'])
            if preprocess:
                pub2ref_df.to_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)),
                                                                                key = 'pub2ref', mode = 'w')

        return pub2ref_df

    @property
    def paa_df(self):
        if self._paa_df is None:
            if self.keep_in_memory:
                self._paa_df = self.load_publicationauthoraffiliation()
            else:
                return self.load_publicationauthoraffiliation()

        return self._paa_df

    def load_publicationauthoraffiliation(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'publicationauthoraffiliation')):
            return load_preprocessed_data('publicationauthoraffiliation', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_publicationauthoraffiliation()

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=10**7):

        pubauthaff_int_columns = [0, 1, 2, 3]
        pubauthaff_str_columns = [4, 5]
        pub_column_names = ['PublicationId', 'AuthorId', 'AffiliationId', 'AuthorSequence',  'OrigAuthorName', 'OrigAffiliationName']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'publicationauthoraffiliation')):
                os.mkdir(os.path.join(self.path2database, 'publicationauthoraffiliation'))

        iref = 0
        ifile = 0
        pubauthaff_info = []
        with open(os.path.join(path2mag, 'RawTXT/mag', 'PaperAuthorAffiliations.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                pubauthaff_info.append([load_int(sline[ip]) for ip in pubauthaff_int_columns] + [sline[ip] if len(sline) > ip else '' for ip in pubauthaff_str_columns ])
                iref += 1

                if preprocess and iref % num_file_lines == 0:
                    pd.DataFrame(pubauthaff_info, columns = pub_column_names).to_hdf(
                        os.path.join(path2mag, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)),
                                                                                key = 'publicationauthoraffiliation', mode = 'w')

                    ifile += 1
                    pubauthaff_info = []


            paa_df = pd.DataFrame(pubauthaff_info, columns = pub_column_names)
            if preprocess:
                paa_df.to_hdf(os.path.join(path2mag, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)),
                                                                            key = 'publicationauthoraffiliation', mode = 'w')
        return paa_df

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

    def load_pub2field(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'pub2field')):
            return load_preprocessed_data('pub2field', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_fields()

    @property
    def fieldinfo_df(self):
        if self._fieldinfo_df is None:
            if self.keep_in_memory:
                self._fieldinfo_df = self.load_fieldinfo()
            else:
                return self.load_fieldinfo()

        return self._fieldinfo_df

    def load_fieldinfo(self, preprocess = True, columns = None, isindict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None):

        if preprocess and os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
            return load_preprocessed_data('fieldinfo', self.path2database, columns, isindict, duplicate_subset, duplicate_keep, dropna)
        else:
            return self.parse_fields()

    def parse_fields(self, preprocess = False, num_file_lines=10**7):

        field2get = [0, 5, 6]
        fieldnames = ['FieldId', 'FieldLevel', 'NumberPublications', 'FieldName']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
                os.mkdir(os.path.join(self.path2database, 'fieldinfo'))

        fieldinfo = []
        with open(os.path.join(path2mag, 'RawTXT/advanced', 'FieldsOfStudy.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                fielddata = [load_int(sline[ip]) for ip in field2get] + [sline[2]]
                fieldinfo.append(fielddata)

        field_df = pd.DataFrame(fieldinfo, columns = fieldnames)
        if preprocess:
            field_df.to_hdf(os.path.join(path2mag, 'fieldinfo', 'fieldinfo.hdf'), key = 'field', mode = 'w')


        # and now do pub2field
        paperfields = [0, 1]
        paperfieldnames = ['PublicationId', 'FieldId']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2field')):
                os.mkdir(os.path.join(self.path2database, 'pub2field'))

        ipaper = 0
        ifile = 0
        fieldinfo = []
        with open(os.path.join(path2mag, 'RawTXT/advanced', 'PaperFieldsOfStudy.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                fielddata = [int(sline[ip]) for ip in paperfields]
                fieldinfo.append(fielddata)
                ipaper += 1

                if preprocess and ipaper % num_file_lines == 0:
                    pd.DataFrame(fieldinfo, columns = paperfieldnames).to_hdf(
                        os.path.join(path2mag, 'pub2field', 'pub2field' + str(ifile) + '.hdf'),
                                                                                key = 'pub2field', mode = 'w')

                    ifile += 1
                    fieldinfo = []

        pub2field_df = pd.DataFrame(fieldinfo, columns = paperfieldnames)
        if preprocess:
            pub2field_df.to_hdf(os.path.join(path2mag, 'pub2field', 'pub2field' + str(ifile) + '.hdf'),
                                                                                key = 'pub2field', mode = 'w')
        return pub2field_df


class WOS(object):
    """
    Base class for Web of Science interface.

    TODO: everything

    """

    def __init__(self, path2database = ''):

        self.path2database = path2database

class Dimensions(object):
    """
    Base class for Dimensions interface.

    TODO: everything

    """

    def __init__(self, path2database = ''):

        self.path2database = path2databases

