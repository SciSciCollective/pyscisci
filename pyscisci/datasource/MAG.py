
import os
import sys
import json
import gzip

import pandas as pd
import numpy as np
from nameparser import HumanName
import unicodedata

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float
from pyscisci.database import BibDataBase



class MAG(BibDataBase):
    """
    Base class for Microsoft Academic Graph interface.

    This is an extension of 'BibDataBase' with processing functions specific to the MAG.
    See 'BibDataBase' in database.py for details of non-MAG specific functions.

    The MAG comes structured into three folders: mag, advanced, nlp.
    Explain downloading etc.

    """

    def __init__(self, path2database = '', keep_in_memory = False, show_progress=True):

        self.path2database = path2database
        self.keep_in_memory = keep_in_memory
        self.show_progress = show_progress

        self._affiliation_df = None
        self._pub_df = None
        self._journal_df = None
        self._author_df = None
        self._pub2year = None
        self._pub2ref_df = None
        self._pub2refnoself_df = None
        self._author2pub_df = None
        self._paa_df = None
        self._pub2field_df = None
        self._fieldinfo_df = None

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = int

    def preprocess(self, dflist = None, show_progress=True):
        """
        Bulk preprocess the MAG raw data.

        Parameters
        ----------
        :param dflist: list, default None
            The list of DataFrames to preprocess.  If None, all MAG DataFrames are preprocessed.

        :param show_progress: bool, default True
            Show progress with processing of the data.

        """
        if dflist is None:
            dflist = ['affiliation', 'author', 'publication', 'reference', 'publicationauthoraffiliation', 'fields']

        if 'affiliation' in dflist:
            self.parse_affiliations(preprocess = True, show_progress=show_progress)

        if 'author' in dflist:
            self.parse_authors(preprocess = True, show_progress=show_progress)

        if 'publication' in dflist:
            self.parse_publications(preprocess = True, show_progress=show_progress)

        if 'reference' in dflist:
            self.parse_references(preprocess = True, show_progress=show_progress)

        if 'publicationauthoraffiliation' in dflist:
            self.parse_publicationauthoraffiliation(preprocess = True, show_progress=show_progress)

        if 'fields' in dflist:
            self.parse_fields(preprocess=True, show_progress=show_progress)

    def download_from_source(self):
        #TODO: Error Raising when database isnt defined
        raise NotImplementedError("ToDo")

    def parse_affiliations(self, preprocess=True, show_progress=True):
        """
        Parse the MAG Affilation raw data.

        Parameters
        ----------
        :param preprocess: bool, default True
            Save the processed data in new DataFrames.

        :param show_progress: bool, default True
            Show progress with processing of the data.


        Returns
        ----------
        DataFrame
            Affiliation DataFrame.
        """

        affil_int_columns = [0, 7, 8]
        affil_str_columns = [3, 4, 5, 6]
        affil_float_columns = [9, 10]

        affil_column_names = ['AffiliationId', 'NumberPublications', 'NumberCitations', 'FullName', 'GridId', 'OfficialPage', 'WikiPage', 'Latitude', 'Longitude']

        file_name = os.path.join(self.path2database, 'mag', 'Affiliations.txt')

        affiliation_info = []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, desc='Affiliations', leave=True, disable=not show_progress) as pbar:
            with open(file_name, 'r') as infile:
                for line in infile:
                    sline = line.replace('\n', '').split('\t')
                    affline = [load_int(sline[i]) for i in affil_int_columns]
                    affline += [sline[i] for i in affil_str_columns]
                    affline += [load_float(sline[i]) for i in affil_float_columns]
                    affiliation_info.append(affline)

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

        aff_df = pd.DataFrame(affiliation_info, columns = affil_column_names)

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'affiliation')):
                os.mkdir(os.path.join(self.path2database, 'affiliation'))
            aff_df.to_hdf(os.path.join(self.path2database, 'affiliation', 'affiliation0.hdf'), key = 'affiliation', mode = 'w')

        return aff_df

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6, show_progress=True):
        """
        Parse the MAG Author raw data.

        Parameters
        ----------
        :param preprocess: bool, default True
            Save the processed data in new DataFrames.

        :param process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.

        :param num_file_lines: int, default 5*10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        :param show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Author DataFrame.
        """

        author_int_columns = [0, 4, 5, 6]

        author_column_names = ['AuthorId', 'LastKnownAffiliationId', 'NumberPublications', 'NumberCitations', 'FullName']
        if process_name:
            author_column_names += ['LastName', 'FirstName', 'MiddleName']


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'author')):
                os.mkdir(os.path.join(self.path2database, 'author'))

        file_name = os.path.join(self.path2database, 'mag', 'Authors.txt')

        iauthor = 0
        ifile = 0
        authorinfo = []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, desc='Authors', disable=not show_progress, leave=True) as pbar:
            with open(file_name, 'r') as infile:

                for line in infile:

                    # split the line and keep only the relevant columns
                    sline = line.split('\t')
                    adata = [load_int(sline[ip]) for ip in author_int_columns] + [sline[2]]

                    # process the first, middle, and last names for the author
                    if process_name:
                        hname = HumanName(unicodedata.normalize('NFD', sline[2]))
                        adata += [hname.last, hname.first, hname.middle]

                    authorinfo.append(adata)
                    iauthor += 1

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

                    # time to save
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

    def parse_publications(self, preprocess = True, num_file_lines=5*10**6, preprocess_dicts = True, show_progress=True):
        """
        Parse the MAG Publication and Journal raw data.

        Parameters
        ----------
        :param preprocess: bool, default True
            Save the processed data in new DataFrames.

        :param num_file_lines: int, default 5*10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        :param preprocess_dicts: bool, default True
            Save the processed Year and DocType data as dictionaries.

        :param show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Publication DataFrame.
        """

        # first do the journal information
        journal_str_col = [2, 4, 5, 6]
        journal_column_names = ['JournalId', 'FullName', 'Issn', 'Publisher', 'Webpage']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'journal')):
                os.mkdir(os.path.join(self.path2database, 'journal'))

        file_name = os.path.join(self.path2database, 'mag', 'Journals.txt')

        journal_info = []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, desc='Journals', leave=True, disable=not show_progress) as pbar:
            with open(os.path.join(self.path2database, 'mag', 'Journals.txt'), 'r') as infile:
                for line in infile:
                    # split the line and keep only the relevant columns
                    sline = line.replace('\n', '').split('\t')
                    jline = [load_int(sline[0])] + [sline[i] for i in journal_str_col]
                    journal_info.append(jline)

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

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

        file_name = os.path.join(self.path2database, 'mag', 'Papers.txt')

        ipub = 0
        ifile = 0
        pubinfo = []

        pub2year = {}
        pub2doctype = {}
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, desc='Publications', leave=True, disable=not show_progress) as pbar:
            with open(file_name, 'r') as infile:
                for line in infile:
                    # split the line and keep only the relevant columns
                    sline = line.replace('\n', '').split('\t')
                    pline = [load_int(sline[ip]) for ip in pub_int_columns] + [sline[ip] for ip in pub_str_columns] + [doctype[sline[3]]]
                    pub2year[pline[0]] = pline[1]
                    if doctype[sline[3]] != '':
                        pub2doctype[pline[0]] = doctype[sline[3]]

                    pubinfo.append(pline)
                    ipub += 1

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

                    if preprocess and ipub % num_file_lines == 0:
                            pd.DataFrame(pubinfo, columns = pub_column_names).to_hdf(
                                os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)),
                                                                                        key = 'publication', mode = 'w')

                            ifile += 1
                            pubinfo = []

            pub_df = pd.DataFrame(pubinfo, columns = pub_column_names)
            if preprocess:
                pub_df.to_hdf(os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)),
                                                                                key = 'publication', mode = 'w')

                if preprocess_dicts:
                    with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
                        outfile.write(json.dumps(pub2year).encode('utf8'))

                    with gzip.open(os.path.join(self.path2database, 'pub2doctype.json.gz'), 'w') as outfile:
                        outfile.write(json.dumps(pub2doctype).encode('utf8'))

        return pub_df


    def parse_references(self, preprocess = False, num_file_lines=10**7, show_progress=True):
        """
        Parse the MAG References raw data.

        Parameters
        ----------
        :param preprocess: bool, default True
            Save the processed data in new DataFrames.

        :param num_file_lines: int, default 10**7
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        :param show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Pub2Ref DataFrame.
        """
        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2ref')):
                os.mkdir(os.path.join(self.path2database, 'pub2ref'))

        file_name = os.path.join(self.path2database, 'mag', 'PaperReferences.txt')

        iref = 0
        ifile = 0
        pub2ref_info = []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, unit_divisor=1024, desc='References', leave=True, disable=not show_progress) as pbar:
            with open(file_name, 'r') as infile:
                for line in infile:
                    # split the line and keep only the relevant columns
                    sline = line.replace('\n', '').split('\t')
                    pub2ref_info.append([load_int(sline[ip]) for ip in range(2)])
                    iref += 1

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

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

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=5*10**6, show_progress=True):
        """
        Parse the MAG PublicationAuthorAffiliation raw data.

        Parameters
        ----------
        :param preprocess: bool, default True
            Save the processed data in new DataFrames.

        :param num_file_lines: int, default 10**7
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        :param show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            PublicationAuthorAffiliation DataFrame.
        """
        pubauthaff_int_columns = [0, 1, 2, 3]
        pubauthaff_str_columns = [4, 5]
        pub_column_names = ['PublicationId', 'AuthorId', 'AffiliationId', 'AuthorSequence',  'OrigAuthorName', 'OrigAffiliationName']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'publicationauthoraffiliation')):
                os.mkdir(os.path.join(self.path2database, 'publicationauthoraffiliation'))

        file_name = os.path.join(self.path2database, 'mag', 'PaperAuthorAffiliations.txt')

        iref = 0
        ifile = 0
        pubauthaff_info = []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, desc='PaperAuthorAffiliations', leave=True, disable=not show_progress) as pbar:
            with open(file_name, 'r') as infile:
                for line in infile:
                    sline = line.replace('\n', '').split('\t')
                    pubauthaff_info.append([load_int(sline[ip]) for ip in pubauthaff_int_columns] + [sline[ip] if len(sline) > ip else '' for ip in pubauthaff_str_columns ])
                    iref += 1

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

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

    def parse_fields(self, preprocess = False, num_file_lines=10**7, show_progress=True):
        """
        Parse the MAG Paper Field raw data.

        Parameters
        ----------
        :param preprocess: bool, default True
            Save the processed data in new DataFrames.

        :param num_file_lines: int, default 10**7
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        :param show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Pub2Field DataFrame.
        """
        field2get = [0, 5, 6]
        fieldnames = ['FieldId', 'FieldLevel', 'NumberPublications', 'FieldName']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
                os.mkdir(os.path.join(self.path2database, 'fieldinfo'))

        fieldinfo = []
        with open(os.path.join(self.path2database, 'advanced', 'FieldsOfStudy.txt'), 'r') as infile:

            for line in infile:
                sline = line.split('\t')
                fielddata = [load_int(sline[ip]) for ip in field2get] + [sline[2]]
                fieldinfo.append(fielddata)

        field_df = pd.DataFrame(fieldinfo, columns = fieldnames)
        if preprocess:
            field_df.to_hdf(os.path.join(self.path2database, 'fieldinfo', 'fieldinfo0.hdf'), key = 'field', mode = 'w')


        # and now do pub2field
        paperfields = [0, 1]
        paperfieldnames = ['PublicationId', 'FieldId']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, 'pub2field')):
                os.mkdir(os.path.join(self.path2database, 'pub2field'))

        file_name = os.path.join(self.path2database, 'advanced', 'PaperFieldsOfStudy.txt')

        ipaper = 0
        ifile = 0
        fieldinfo = []
        with tqdm(total=os.path.getsize(file_name), unit='iB', unit_scale=True, desc='PaperFieldsOfStudy', leave=True, disable=not show_progress) as pbar:
            with open(file_name, 'r') as infile:

                for line in infile:
                    sline = line.split('\t')
                    fielddata = [int(sline[ip]) for ip in paperfields]
                    fieldinfo.append(fielddata)
                    ipaper += 1

                    # update progress bar
                    pbar.update(sys.getsizeof(line))

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


