import os
import sys
import json
import gzip

import pandas as pd
import numpy as np
from nameparser import HumanName
import requests
from lxml import etree
from io import BytesIO

# determine if we are loading from a jupyter notebook (to make pretty progress bars)
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float, load_html_str
from pyscisci.database import BibDataBase

class DBLP(BibDataBase):
    """
    Base class for DBLP interface.

    The DBLP comes as a single xml file.  It can be downloaded from [DBLP](https://dblp.uni-trier.de/) via `donwload_from_source`

    There is no citation information!

    """


    def __init__(self, path2database = '', database_extension='csv.gz', keep_in_memory = False, global_filter=None, 
        enable_dask=False, show_progress=True):

        self._default_init(path2database, database_extension, keep_in_memory, global_filter, enable_dask, show_progress)

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = str
        self.JournalIdType=str

        self.path2journal = self.path2pub
        self.path2paa = 'publicationauthor'

    def _blank_dblp_publication(self, PublicationId = 0):
        record = {}
        record['PublicationId'] = PublicationId
        record['Title'] = ''
        record['Year'] = 0
        record['Volume'] = 0
        record['Number'] = ''
        record['Pages'] = ''
        record['JournalId'] = ''
        record['EE'] = ''
        record['TeamSize'] = 0
        record['Month'] = 1
        record['DocType'] = ''
        return record

    def _save_dataframes(self, ifile, publication, author, author_columns, author2pub):

        publication = pd.DataFrame(publication)
        publication['PublicationId'] = publication['PublicationId'].astype(int)
        publication['Year'] = publication['Year'].astype(int)
        publication['Volume'] = pd.to_numeric(publication['Volume'])
        publication['TeamSize'] = publication['TeamSize'].astype(int)
        fname = os.path.join(self.path2database, self.path2pub, '{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
        self.save_data_file(publication, fname, key =self.path2pub)


        author = pd.DataFrame(author, columns = author_columns)
        author['AuthorId'] = author['AuthorId'].astype(int)
        fname = os.path.join(self.path2database, self.path2author, '{}{}.{}'.format(self.path2author, ifile, self.database_extension))
        self.save_data_file(author, fname, key =self.path2author)

        author2pub = pd.DataFrame(author2pub, columns = ['PublicationId', 'AuthorId', 'AuthorSequence'], dtype=int)
        fname = os.path.join(self.path2database, self.path2paa, '{}{}.{}'.format(self.path2paa, ifile, self.database_extension))
        self.save_data_file(author2pub, fname, key =self.path2paa)

    def preprocess(self, xml_file_name = 'dblp.xml.gz', process_name=True, num_file_lines=10**6, show_progress=True):
        """
        Bulk preprocess of the DBLP raw data.

        Parameters
        ----------
        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.

        xml_file_name: str, default 'dblp.xml.gz'
            The xml file name.

        num_file_lines: int, default 10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        show_progress: bool, default True
            Show progress with processing of the data.

        """

        ACCEPT_DOCTYPES = set(['article', 'inproceedings', 'proceedings', 'book', 'incollection', 'phdthesis', 'mastersthesis'])
        REJECT_DOCTYPES = set(['www'])
        DATA_ITEMS = ['title', 'booktitle', 'year', 'journal', 'ee',' url', 'month', 'mdate', 'isbn', 'publisher']
        SKIP_FIELDS = ['note', 'cite', 'cdrom', 'crossref', 'editor',  'series', 'tt', 'school', 'chapter', 'address']

        doctype = {'Article': 'j', 'Book Review':'br', 'Letter':"l", 'Review':"rev", 'Correction':"corr",
       'Editorial Material':"editorial", 'News Item':"news", 'Bibliography':"bib",
       'Biographical-Item':"bibi", 'Hardware Review':"hr", 'Meeting Abstract':"ma", 'Note':"note",
       'Discussion':"disc", 'Item About an Individual':"indv", 'Correction, Addition':"corradd",
       'Chronology':"chrono", 'Software Review':"sr", 'Reprint':"re", 'Database Review':"dr", 
       "Journal":'j', 'Book':'b', '':'', 'BookChapter':'bc', 'Conference':'c', 'Dataset':'d', 'Patent':'p', 'Repository':'r', 'Thesis':'t',
       'article': 'j', 'book':'b', '':'', 'phdthesis':'t', 'proceedings':'c', 'inproceedings':'c',
        'mastersthesis':'mst', 'incollection':'coll'}

        html_format_keys = ['<sub>', '</sub>', '<sup>', '</sup>', '<i>', '</i>']

        if show_progress:
            print("Starting to preprocess the DBLP database.")

        for hier_dir_type in [self.path2pub, self.path2author, self.path2paa]:

            if not os.path.exists(os.path.join(self.path2database, hier_dir_type)):
                os.mkdir(os.path.join(self.path2database, hier_dir_type))

        publication = []
        author = []
        author2pub = []
        journal = []

        PublicationId = 1
        AuthorId = 1
        aname2aid = {}
        author_columns = ['AuthorId', 'FullName']
        if process_name:
            author_columns += ['LastName', 'FirstName', 'MiddleName']
        JournalId = 1
        jname2jid = {}


        pub_record = self._blank_dblp_publication(PublicationId)
        pub_authors = []
        AuthorCount = 0

        ifile = 0

        # read dtd - this takes
        path2database = self.path2database # remove self to use inside of this class
        class DTDResolver(etree.Resolver):
            def resolve(self, system_url, public_id, context):
                return self.resolve_filename(os.path.join(path2database, system_url), context)

        if '.gz' in xml_file_name:
            with gzip.open(os.path.join(self.path2database, xml_file_name), 'r') as infile:
                xml_file = infile.read()

        else:
            with open(os.path.join(self.path2database, xml_file_name), 'r') as infile:
                xml_file = infile.read().encode('latin1')

        # extract the desired fields from the XML tree  #
        bytesxml = BytesIO(xml_file)
        xmltree = etree.iterparse(bytesxml, load_dtd=True, resolve_entities=True)
        xmltree.resolvers.add(DTDResolver())

        if show_progress:
            print("Xml tree parsed, iterating through elements.")

        last_position = 0
        xml_size = bytesxml.getbuffer().nbytes
        with tqdm(total=xml_size, unit='iB', unit_scale=True, desc='dblp.xml', leave=True, disable=not show_progress) as pbar:
            for event, elem in xmltree:
                if elem.tag == 'title' or elem.tag == 'booktitle':
                    pub_record['Title'] = load_html_str(elem.text)

                elif elem.tag == 'year':
                    pub_record['Year'] = load_int(elem.text)

                elif elem.tag == 'month':
                    pub_record['Month'] = load_int(elem.text)

                elif elem.tag == 'volume':
                    pub_record['Volume'] = load_int(elem.text)

                elif elem.tag == 'number':
                    pub_record['Number'] = load_html_str(elem.text)

                elif elem.tag == 'pages':
                    pub_record['Pages'] = load_html_str(elem.text)

                elif elem.tag == 'journal':
                    pub_record['JournalId'] = load_html_str(elem.text)

                elif elem.tag == 'url':
                    pub_record['URL'] = load_html_str(elem.text)

                elif elem.tag == 'ee':
                    pub_record['EE'] = load_html_str(elem.text)

                elif elem.tag == 'author':
                    AuthorCount += 1
                    fullname = load_html_str(elem.text)
                    if aname2aid.get(fullname, None) is None:
                        if process_name:
                            fullname = ''.join([i for i in fullname if not i.isdigit()]).strip()
                            hname = HumanName(fullname)
                            author.append([AuthorId, fullname, hname.last, hname.first, hname.middle])
                        else:
                            author.append([AuthorId, fullname])
                        aname2aid[fullname] = AuthorId
                        AuthorId += 1

                    pub_authors.append([PublicationId, aname2aid[fullname], AuthorCount])

                elif elem.tag in ACCEPT_DOCTYPES:
                    pub_record['TeamSize'] = AuthorCount
                    pub_record['DocType'] = doctype[load_html_str(elem.tag)]

                    publication.append(pub_record)
                    author2pub.extend(pub_authors)
                    PublicationId += 1
                    pub_record = self._blank_dblp_publication(PublicationId)
                    AuthorCount = 0
                    pub_authors = []

                    # update progress bar
                    pbar.update(bytesxml.tell() - last_position)
                    last_position = bytesxml.tell()

                    if num_file_lines > 0 and (PublicationId % num_file_lines) == 0:

                        self._save_dataframes(ifile, publication, author, author_columns, author2pub)

                        ifile += 1

                        publication = []
                        author = []
                        author2pub = []

                elif elem.tag in REJECT_DOCTYPES:
                    # the record was from a rejected category so reset record
                    pub_record = self._blank_dblp_publication(PublicationId)
                    AuthorCount = 0
                    pub_authors = []

                elif elem.tag in SKIP_FIELDS:
                    pass

        del xmltree

        self._save_dataframes(ifile, publication, author, author_columns, author2pub)




    def download_from_source(self, source_url='https://dblp.uni-trier.de/xml/', xml_file_name = 'dblp.xml.gz',
        dtd_file_name = 'dblp.dtd', show_progress=True):
        """
        Download the DBLP raw xml file and the dtd formating information from [DBLP](https://dblp.uni-trier.de/).
            1. dblp.xml.gz - the compressed xml file
            2. dblp.dtd - the dtd containing xml syntax

        The files will be saved to the path specified by `path2database`.

        Parameters
        ----------
        source_url: str, default 'https://dblp.uni-trier.de/xml/'
            The base url from which to download.

        xml_file_name: str, default 'dblp.xml.gz'
            The xml file name.

        dtd_file_name: str, default 'dblp.dtd'
            The dtd file name.

        show_progress: bool, default True
            Show progress with processing of the data.

        """

        block_size = 1024 #1 Kibibyte

        req_stream = requests.get(os.path.join(source_url, xml_file_name), stream=True)
        total_size = int(req_stream.headers.get('content-length', 0))

        if not os.path.exists(self.path2database):
            os.mkdir(self.path2database)

        with tqdm(total=total_size, unit='iB', unit_scale=True, desc='dblp.xml.gz', leave=True, disable=not show_progress) as pbar:
            with open(os.path.join(self.path2database, xml_file_name), "wb") as outfile:

                for block in req_stream.iter_content(block_size):
                    outfile.write(block)

                    # update progress bar
                    pbar.update(len(block))

        with open(os.path.join(self.path2database, dtd_file_name), 'w') as outfile:
            outfile.write(requests.get(os.path.join(source_url, dtd_file_name)).content.decode('latin1'))

    def parse_affiliations(self, preprocess = False):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_authors(self, preprocess = False, process_name = True, num_file_lines = 5*10**6):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_publications(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_references(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP does not contain reference or citation information.")

    def parse_publicationauthoraffiliation(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def parse_fields(self, preprocess = False, num_file_lines=10**7):
        raise NotImplementedError("DBLP does not contain field information.")

    @property
    def author2pub(self):
        """
        The DataFrame keeping all publication, author relationships.  Columns may depend on the specific datasource.

        Notes
        -------
        columns: 'PublicationId', 'AuthorId', 'AuthorOrder'

        """
        if self._author2pub is None:
            if self.keep_in_memory:
                self._author2pub = self.load_publicationauthor(show_progress=self.show_progress)
            else:
                return self.load_publicationauthor(show_progress=self.show_progress)

        return self._author2pub

    def load_publicationauthor(self, preprocess = True, columns = None, filter_dict = None, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, show_progress=False):
        """
        Load the PublicationAuthor DataFrame from a preprocessed directory.  For DBLP, you must run preprocess before
        the dataframe is available for use.

        Parameters
        ----------
        preprocess : bool, default True, Optional
            Attempt to load from the preprocessed directory.

        columns : list, default None, Optional
            Load only this subset of columns

        filter_dict : dict, default None, Optional
            Dictionary of format {"ColumnName":"ListofValues"} where "ColumnName" is a data column
            and "ListofValues" is a sorted list of valid values.  A DataFrame only containing rows that appear in
            "ListofValues" will be returned.

        duplicate_subset : list, default None, Optional
            Drop any duplicate entries as specified by this subset of columns

        duplicate_keep : str, default 'last', Optional
            If duplicates are being dropped, keep the 'first' or 'last'
            (see `pandas.DataFram.drop_duplicates <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop_duplicates.html>`_)

        dropna : list, default None, Optional
            Drop any NaN entries as specified by this subset of columns

        Returns
        -------
        DataFrame
            PublicationAuthor DataFrame.

        """
        if show_progress:
            show_progress='Loading PublicationAuthor'
        if preprocess and os.path.exists(os.path.join(self.path2database, 'publicationauthor')):
            return load_preprocessed_data('publicationauthor', path2database=self.path2database, columns=columns,
                filter_dict=filter_dict, duplicate_subset=duplicate_subset, duplicate_keep=duplicate_keep, dropna=dropna,
                show_progress=show_progress)
        else:
            raise NotImplementedError("DBLP is stored as a single xml file.  Run preprocess to parse the file.")

    def load_journals(self, preprocess = True, columns = None, filter_dict = {}, duplicate_subset = None,
        duplicate_keep = 'last', dropna = None, prefunc2apply=None, postfunc2apply=None, show_progress=False):

        raise NotImplementedError("The DBLP does not have prespecified journal information.")
