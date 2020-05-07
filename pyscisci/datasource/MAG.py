
import os
import json
import gzip

import pandas as pd
import numpy as np
from nameparser import HumanName

from readwrite import load_preprocessed_data
from pyscisci.database import BibDataBase

class MAG(BibDataBase):
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
        self._pub2refnoself_df = None
        self._author2pub_df = None
        self._paa_df = None

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = int

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

    def parse_publications(self, preprocess = False, num_file_lines=10**7):

        # first do the journal information
        journal_str_col = [2, 4, 5, 6]
        journal_column_names = ['JournalId', 'FullName', 'Issn', 'Publisher', 'Webpage']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'journal')):
                os.mkdir(os.path.join(self.path2database, 'journal'))

        journal_info = []
        with open(os.path.join(self.path2database, 'RawTXT/mag', 'Journals.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                jline = [load_int(sline[0])] + [sline[i] for i in journal_str_col]
                journal_info.append(jline)

        journal_df = pd.DataFrame(journal_info, columns = journal_column_names)
        if preprocess:
            journal_df.to_hdf(os.path.join(self.path2database, 'journal', 'journal.hdf'), key = 'journal', mode = 'w')

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
        with open(os.path.join(self.path2database, 'RawTXT/mag', 'PaperAuthorAffiliations.txt'), 'r') as infile:
            for line in infile:
                sline = line.replace('\n', '').split('\t')
                pubauthaff_info.append([load_int(sline[ip]) for ip in pubauthaff_int_columns] + [sline[ip] if len(sline) > ip else '' for ip in pubauthaff_str_columns ])
                iref += 1

                if preprocess and iref % num_file_lines == 0:
                    pd.DataFrame(pubauthaff_info, columns = pub_column_names).to_hdf(
                        os.path.join(self.path2database, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)),
                                                                                key = 'publicationauthoraffiliation', mode = 'w')

                    ifile += 1
                    pubauthaff_info = []


            paa_df = pd.DataFrame(pubauthaff_info, columns = pub_column_names)
            if preprocess:
                paa_df.to_hdf(os.path.join(self.path2database, 'publicationauthoraffiliation', 'publicationauthoraffiliation{}.hdf'.format(ifile)),
                                                                            key = 'publicationauthoraffiliation', mode = 'w')
        return paa_df

    def parse_fields(self, preprocess = False, num_file_lines=10**7):

        field2get = [0, 5, 6]
        fieldnames = ['FieldId', 'FieldLevel', 'NumberPublications', 'FieldName']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
                os.mkdir(os.path.join(self.path2database, 'fieldinfo'))

        fieldinfo = []
        with open(os.path.join(self.path2database, 'RawTXT/advanced', 'FieldsOfStudy.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                fielddata = [load_int(sline[ip]) for ip in field2get] + [sline[2]]
                fieldinfo.append(fielddata)

        field_df = pd.DataFrame(fieldinfo, columns = fieldnames)
        if preprocess:
            field_df.to_hdf(os.path.join(self.path2database, 'fieldinfo', 'fieldinfo.hdf'), key = 'field', mode = 'w')


        # and now do pub2field
        paperfields = [0, 1]
        paperfieldnames = ['PublicationId', 'FieldId']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2field')):
                os.mkdir(os.path.join(self.path2database, 'pub2field'))

        ipaper = 0
        ifile = 0
        fieldinfo = []
        with open(os.path.join(self.path2database, 'RawTXT/advanced', 'PaperFieldsOfStudy.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                fielddata = [int(sline[ip]) for ip in paperfields]
                fieldinfo.append(fielddata)
                ipaper += 1

                if preprocess and ipaper % num_file_lines == 0:
                    pd.DataFrame(fieldinfo, columns = paperfieldnames).to_hdf(
                        os.path.join(self.path2database, 'pub2field', 'pub2field' + str(ifile) + '.hdf'),
                                                                                key = 'pub2field', mode = 'w')

                    ifile += 1
                    fieldinfo = []

        pub2field_df = pd.DataFrame(fieldinfo, columns = paperfieldnames)
        if preprocess:
            pub2field_df.to_hdf(os.path.join(self.path2database, 'pub2field', 'pub2field' + str(ifile) + '.hdf'),
                                                                                key = 'pub2field', mode = 'w')
        return pub2field_df


