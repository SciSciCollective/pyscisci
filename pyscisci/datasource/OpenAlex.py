
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

from pyscisci.datasource.readwrite import load_preprocessed_data, load_int, load_float, load_bool
from pyscisci.database import BibDataBase

openalex_works_dfset = {'publications', 'references', 'publicationauthoraffiliation', 'concepts', 'fields', 'abstracts'}
openalex_dataframe_set = {'publications', 'works', 'venues', 'authors', 'institutions', 'affiliations', 'concepts', 'fields'}

class OpenAlex(BibDataBase):
    """
    Base class for the OpenAlex interface.

    This is an extension of 'BibDataBase' with processing functions specific to the MAG.
    See 'BibDataBase' in database.py for details of non-MAG specific functions.

    The MAG comes structured into three folders: mag, advanced, nlp.
    Explain downloading etc.

    """

    def __init__(self, path2database = '', database_extension='csv.gz', keep_in_memory = False, global_filter=None, 
        enable_dask=False, show_progress=True):

        self._default_init(path2database, database_extension, keep_in_memory, global_filter, enable_dask, show_progress)

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = int
        self.JournalIdType = int

    def preprocess(self, dataframe_list = None, show_progress=True):
        """
        Bulk preprocess the MAG raw data.

        Parameters
        ----------
        dataframe_list: list, default None
            The list of DataFrames to preprocess.  If None, all MAG DataFrames are preprocessed.
            
        show_progress: bool, default True
            Show progress with processing of the data.

        """
        if dataframe_list is None or 'all' in dataframe_list:
            dataframe_list = openalex_works_dfset.union(openalex_dataframe_set)

        pubdataframe_list = set(dataframe_list).intersection(openalex_works_dfset)

        if 'affiliations' in dataframe_list or 'institutions' in dataframe_list:
            self.parse_affiliations(preprocess = True, show_progress=show_progress)

        if 'authors' in dataframe_list:
            self.parse_authors(preprocess = True, show_progress=show_progress)

        if 'publications' in dataframe_list or 'works' in dataframe_list:
            self.parse_publications(preprocess = True, dataframe_list=pubdataframe_list, show_progress=show_progress)

        if 'concepts' in dataframe_list or 'fields' in dataframe_list:
            self.parse_concepts(preprocess=True, show_progress=show_progress)

        if 'venues' in dataframe_list or 'journals' in dataframe_list:
            self.parse_venues(preprocess=True, show_progress=show_progress)

    def download_from_source(self, aws_access_key_id='', aws_secret_access_key='', specific_update='', aws_bucket = 'openalex',
        dataframe_list = 'all', rewrite_existing = False, edit_works = True, show_progress=True):

        """
        Download the OpenAlex files from Amazon Web Services.
        The files will be saved to the path specified by `path2database` into RawXML.

        Parameters
        ----------
        source_url: str, default 'ftp.ncbi.nlm.nih.gov'
            The base url for the ftp server from which to download.
        
        aws_access_key_id: str
            The acess key for AWS (not required)

        aws_secret_access_key: str
            The secret acess key for AWS (not required)

        specific_update: str
            Download only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is downloaded.

        dataframe_list: list
            The data types to download and save from OpenAlex.
                'all'
                'affiliations'
                'authors'
                'publications'
                'references'
                'publicationauthoraffiliation'
                'fields'
                'abstracts'

        rewrite_existing: bool, default False
            If True, overwrite existing files or if False, only download any missing files.

        edit_works: bool, default True
            If True, edit the works to remove pieces of data.  If False, force keeping all entries from the works file.

        show_progress: bool, default True
            Show progress with processing of the data.
        
        # edited from https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
        """

        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
        except ImportError:
            raise ImportError("To download the OpenAlex data, you must install the aws S3 client and python package boto3.")

        if len(aws_access_key_id) > 0 and len(aws_secret_access_key) > 0:
            s3_client = boto3.client(aws_access_key_id=aws_access_key_id,
              aws_secret_access_key=aws_secret_access_key)
        else:
            s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))


        dataframe_list = set(dataframe_list)

        if dataframe_list is None or 'all' in dataframe_list:
            dataframe_list = dataframe_list.union(openalex_works_dfset).union(openalex_dataframe_set)
        else:
            # some dataframes have different names from those in pyscisci
            if 'publications' in dataframe_list: dataframe_list.add('works')
            if 'journals' in dataframe_list: dataframe_list.add('venues')
            if 'fields' in dataframe_list: dataframe_list.add('concepts')
            if 'affiliations' in dataframe_list: dataframe_list.add('institutions')

        # see if we have to do any editing
        if edit_works and any(not pubsub in dataframe_list for pubsub in ['works', 'references', 'publicationauthoraffiliation', 'concepts', 'abstracts']):
            edit_works = True
        else:
            edit_works = False
        

        
        if not rewrite_existing:
            files_already_downloaded = [os.path.join(dirpath.replace(self.path2database, ''), file) for (dirpath, dirnames, filenames) in os.walk(self.path2database) for file in filenames]
        else:
            files_already_downloaded = []

        keys = []
        dirs = []
        next_token = ''
        
        base_kwargs = {'Bucket':aws_bucket}
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != '':
                kwargs.update({'ContinuationToken': next_token})
            results = s3_client.list_objects_v2(**kwargs)
            
            for cdict in results.get('Contents', {}):
                k = cdict.get('Key')

                if not k in files_already_downloaded:
                    if 'data' in k: 
                        if k.split('/')[1] in dataframe_list and (specific_update=='' or k.split('/')[2] == 'updated_date={}'.format(specific_update)):
                            if k[-1] != '/':
                                keys.append(k)
                            else:
                                dirs.append(k)
                    else:
                        keys.append(k)
            next_token = results.get('NextContinuationToken')

        # make all of the directories
        for d in dirs:
            dest_pathname = os.path.join(self.path2database, d)
            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
        
        # now download all of the 
        for k in tqdm(keys, desc='Downloading OpenAlex File Number', leave=True, disable=not show_progress):
            dest_pathname = os.path.join(self.path2database, k)

            if not os.path.exists(os.path.dirname(dest_pathname)):
                os.makedirs(os.path.dirname(dest_pathname))
            
            s3_client.download_file(aws_bucket, k, dest_pathname)

            # the works contains references, concepts, and astracts, check if we keep these
            if edit_works and (len(k.split('/')) > 1 and k.split('/')[1] == 'works' and k[-3:]=='.gz'):
                with gzip.open(dest_pathname, 'r') as infile:
                    with gzip.open(dest_pathname.split('.')[0] + '_temp.gz', 'w') as outfile:
                        for line in infile:
                            if line != '\n'.encode():
                                jline = json.loads(line)

                                if ( not ('fields' in dataframe_list or 'concepts' in dataframe_list) ) and 'concepts' in jline:
                                    del jline['concepts']

                                if ( not ('references' in dataframe_list) ) and 'referenced_works' in jline:
                                    del jline['referenced_works']

                                if ( not ('publicationauthoraffiliation' in dataframe_list) ) and 'authorships' in jline:
                                    del jline['authorships']

                                if ( not ('abstracts' in dataframe_list) ) and 'abstract_inverted_index' in jline:
                                    del jline['abstract_inverted_index']

                                newline = json.dumps(jline) + "\n"
                                outfile.write(newline.encode('utf-8'))

                if os.path.exists(dest_pathname):
                    os.remove(dest_pathname)
                os.rename(dest_pathname.split('.')[0] + '_temp.gz', dest_pathname)



    def clean_openalex_ids(self, oid):
        try:
            oid = oid.replace('https://openalex.org/', '')
            return int(oid[1:])
        except:
            return None

    def parse_affiliations(self, preprocess=True, specific_update='', show_progress=True):
        """
        Parse the OpenAlex Affilation raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.


        Returns
        ----------
        DataFrame
            Affiliation DataFrame.
        """

        affil_column_names = ['AffiliationId', 'FullName', 'GridId', 'WikiPage', 'Country', 'City', 'Region', 'Latitude', 'Longitude', 'NumberPublications', 'NumberCitations']

        if specific_update == '' or specific_update is None:
            institution_dir = os.path.join(self.path2database, 'data', 'institutions')
        else:
            institution_dir = os.path.join(self.path2database, 'data/institutions', 'updated_date={}'.format(specific_update))

        if not os.path.exists(institution_dir):
            raise ValueError("The institutions data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(institution_dir) for file in filenames if '.gz' in file]


        if not os.path.exists(os.path.join(self.path2database, self.path2affiliation)):
            os.mkdir(os.path.join(self.path2database, self.path2affiliation))

        ifile = 0
        affiliation_info = []
        for file_name in tqdm(files_to_parse, desc='Affiliations', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(institution_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        aline = json.loads(line)
                        affline = [self.clean_openalex_ids(aline['id']), aline.get('display_name', None)]
                        otherids = aline.get('ids', {})
                        affline += [otherids.get(idname, None) for idname in ['grid', 'wikipedia']]
                        geoinfo = aline.get('geo', {})
                        affline += [geoinfo.get(idname, None) for idname in ['country_code', 'city', 'region']]
                        affline += [load_float(geoinfo.get(idname, None)) for idname in ['latitude', 'longitude']]
                        affline += [load_int(aline.get(idname, None)) for idname in ['works_count', 'cited_by_count']]
                        affiliation_info.append(affline)
                    

            aff = pd.DataFrame(affiliation_info, columns = affil_column_names)
            if preprocess:
                fname = os.path.join(self.path2database, self.path2affiliation, '{}{}.{}'.format(self.path2affiliation, ifile, self.database_extension))
                self.save_data_file(aff, fname, key =self.path2affiliation)

                ifile += 1
                affiliation_info = []

        return aff

    def parse_authors(self, preprocess = True, specific_update='', process_name = True, show_progress=True):
        """
        Parse the OpenAlex Author raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        process_name: bool, default True
            If True, then when processing the raw file, the package `NameParser <https://nameparser.readthedocs.io/en/latest/>`_
            will be used to split author FullNames.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Author DataFrame.
        """

        if specific_update == '' or specific_update is None:
            author_dir = os.path.join(self.path2database, 'data', 'authors')
        else:
            author_dir = os.path.join(self.path2database, 'data/authors', 'updated_date={}'.format(specific_update))

        if not os.path.exists(author_dir):
            raise ValueError("The authors data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(author_dir) for file in filenames if '.gz' in file]


        author_column_names = ['AuthorId', 'FullName', 'LastKnownAffiliationId', 'ORCID', 'WikiPage', 'NumberPublications', 'NumberCitations']
        if process_name:
            author_column_names += ['LastName', 'FirstName', 'MiddleName']


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2author)):
                os.mkdir(os.path.join(self.path2database, self.path2author))

        ifile = 0
        authorinfo = []
        author = pd.DataFrame(authorinfo, columns = author_column_names)
        for file_name in tqdm(files_to_parse, desc='Authors', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(author_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        aline = json.loads(line)
                        authline = [self.clean_openalex_ids(aline['id']), aline.get('display_name', None), self.clean_openalex_ids(aline.get('last_known_institution', None))]
                        otherids = aline.get('ids', {})
                        authline += [otherids.get(idname, None) for idname in ['orcid', 'wikipedia']]
                        authline += [load_int(aline.get(idname, None)) for idname in ['works_count', 'cited_by_count']]

                        # process the first, middle, and last names for the author
                        if process_name:
                            name = aline.get('display_name', '')
                            if name is None:
                                name = ''
                            hname = HumanName(unicodedata.normalize('NFD', name))
                            authline += [hname.last, hname.first, hname.middle]

                        authorinfo.append(authline)

                author = pd.DataFrame(authorinfo, columns = author_column_names)

                # time to save
                if preprocess:
                    fname = os.path.join(self.path2database, self.path2author, '{}{}.{}'.format(self.path2author, ifile, self.database_extension))
                    self.save_data_file(author, fname, key =self.path2author)

                    ifile += 1
                    authorinfo = []

        return author

    def parse_venues(self, preprocess = True, specific_update='', show_progress=True):
        """
        Parse the OpenAlex Venues raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Venue DataFrame.
        """
        if specific_update == '' or specific_update is None:
            venue_dir = os.path.join(self.path2database, 'data', 'venues')
        else:
            venue_dir = os.path.join(self.path2database, 'data/venues', 'updated_date={}'.format(specific_update))

        if not os.path.exists(venue_dir):
            raise ValueError("The venues data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(venue_dir) for file in filenames if '.gz' in file]


        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2journal)):
                os.mkdir(os.path.join(self.path2database, self.path2journal))


        venue_column_names = ['VenueId', 'FullName', 'ISSN', 'ISSN_l', 'HomePage', 'NumberPublications', 'NumberCitations']
        venue_info = []
        ifile = 0
        for file_name in tqdm(files_to_parse, desc='Venues', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(venue_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        vline = json.loads(line)
                        vdata = [self.clean_openalex_ids(vline['id'])]
                        vdata += [vline.get(idname, None) for idname in ['display_name', 'issn', 'issn_l', 'homepage_url']]
                        vdata += [load_int(vline.get(idname, None)) for idname in ['works_count', 'cited_by_count']]
                        venue_info.append(vdata)

                venue = pd.DataFrame(venue_info, columns = venue_column_names)
                if preprocess:
                    fname = os.path.join(self.path2database, self.path2journal, '{}{}.{}'.format(self.path2journal, ifile, self.database_extension))
                    self.save_data_file(venue, fname, key =self.path2journal)

                    ifile += 1
                    venue_info = []
        
        return venue


    def parse_publications(self, preprocess = True, specific_update='', preprocess_dicts = True, num_file_lines=10**6,
        dataframe_list = ['publications', 'references', 'publicationauthoraffiliation', 'fields'],
        show_progress=True):
        """
        Parse the OpenAlex Works raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        parse_venues: bool, default True
            Also parse the venue information.

        preprocess_dicts: bool, default True
            Save the processed Year and DocType data as dictionaries.

        num_file_lines: int, default 10**6
            The processed data will be saved into smaller DataFrames, each with `num_file_lines` rows.

        dataframe_list: list
            The data types to download and save from OpenAlex.
                'all'
                'publications'
                'references'
                'publicationauthoraffiliation'
                'fields'
                'abstracts'

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            Publication DataFrame.
        """

        if specific_update == '' or specific_update is None:
            work_dir = os.path.join(self.path2database, 'data', 'works')
        else:
            work_dir = os.path.join(self.path2database, 'data/works', 'updated_date={}'.format(specific_update))

        if not os.path.exists(work_dir):
            raise ValueError("The works data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(work_dir) for file in filenames if '.gz' in file]


        if 'all' in dataframe_list:
            dataframe_list = openalex_works_dfset


        pub_column_names = ['PublicationId', 'JournalId', 'Year', 'NumberCitations', 'Title', 'Date', 'DocType', 'Doi', 'PMID', 'Volume', 'Issue', 'FirstPage', 'LastPage', 'IsRetracted', 'IsParaText']

        if preprocess and (('publications' in dataframe_list) or ('works' in dataframe_list)):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub)):
                os.mkdir(os.path.join(self.path2database, self.path2pub))

            pubinfo = []
            pub2year = {}
            pub2doctype = {}

        if preprocess and ('references' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2ref)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2ref))

            pub2ref = []

        paa_column_names = ['PublicationId', 'AuthorId', 'AffiliationId', 'AuthorSequence', 'OrigAuthorName']

        if preprocess and ('publicationauthoraffiliation' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2paa)):
                os.mkdir(os.path.join(self.path2database, self.path2paa))

            paa = []

        pub2field_column_names = ['PublicationId', 'FieldId', 'FieldLevel', 'Score']

        if preprocess and (('fields' in dataframe_list) or ('concepts' in dataframe_list)):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2field)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2field))

            pub2field = []

        if preprocess and ('abstracts' in dataframe_list):
            if not os.path.exists(os.path.join(self.path2database, self.path2pub2abstract)):
                os.mkdir(os.path.join(self.path2database, self.path2pub2abstract))

            pub2abstract = []

        ifile = 0
        for file_name in tqdm(files_to_parse, desc='Works', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(work_dir, file_name), 'r') as infile:
                
                iline = 0
                
                for line in infile:
                    if line != '\n'.encode():
                        wline = json.loads(line)
                        
                        ownid = self.clean_openalex_ids(wline['id'])

                        if ('publications' in dataframe_list) or ('works' in dataframe_list):
                            wdata = [ownid, self.clean_openalex_ids(wline.get('host_venue', {}).get('id', None))]
                            wdata += [load_int(wline.get(idname, None)) for idname in ['publication_year', 'cited_by_count']]
                            wdata += [wline.get(idname, None) for idname in ['title', 'publication_date', 'type']]
                            doi = wline.get('doi', None)
                            if isinstance(doi, str):
                                doi = doi.replace("https://doi.org/", "")
                            wdata += [doi]
                            pmid = wline.get("ids", {}).get('pmid', None)
                            if isinstance(pmid, str):
                                pmid = pmid.replace("https://pubmed.ncbi.nlm.nih.gov/", "")
                            wdata += [pmid]
                            bibinfo = wline.get('biblio', {})
                            wdata += [bibinfo.get(idname, None) for idname in ['volume', 'issue', 'first_page', 'last_page']]
                            wdata += [load_bool(wline.get(extraid, None)) for extraid in ['is_retracted', 'is_paratext']]

                            if preprocess_dicts:
                                pub2year[ownid] = wdata[2]
                                pub2doctype[ownid] = wdata[7]

                            pubinfo.append(wdata)

                        if ('references' in dataframe_list):
                            pub2ref.extend([[ownid, self.clean_openalex_ids(citedid)] for citedid in wline.get('referenced_works', [])])


                        if ('publicationauthoraffiliation' in dataframe_list):
                            iauthor = 1
                            for authorinfo in wline.get('authorships', []):
                                authorid = self.clean_openalex_ids(authorinfo.get('author', {}).get('id', None))
                                authorname = authorinfo.get('author', {}).get('display_name', '')
                                
                                if (iauthor == 1 and authorinfo.get('author_position', None) != 'first'):
                                    iauthor = None
                                    #print(wline.get('authorships', []))
                                    #raise ValueError('author position')

                                institution_list = authorinfo.get('institutions', [])
                                if len(institution_list) == 0:
                                    institution_list = [None]
                                else:
                                    institution_list = [self.clean_openalex_ids(affinfo.get('id', None)) for affinfo in institution_list]

                                for inid in institution_list:
                                    paa.append([ownid, authorid, inid, iauthor, authorname])

                                if not iauthor is None:
                                    iauthor += 1

                        if ('fields' in dataframe_list) or ('concepts' in dataframe_list):
                            pub_concepts = wline.get('concepts', [])
                            for con_dict in pub_concepts:
                                pub2field.append([ownid, self.clean_openalex_ids(con_dict.get('id', None)), load_int(con_dict.get('level', None)), load_float(con_dict.get('score', None))])
                        

                        if ('abstracts' in dataframe_list) and 'abstract_inverted_index' in wline:
                            pub2abstract.append([ownid, wline['abstract_inverted_index']])

                        iline += 1

                        if iline % num_file_lines == 0:
                            if ('publications' in dataframe_list) or ('works' in dataframe_list):
                                pub = pd.DataFrame(pubinfo, columns = pub_column_names)
                                if preprocess:
                                    fname = os.path.join(self.path2database, self.path2pub, '{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
                                    self.save_data_file(pub, fname, key =self.path2pub)
                                    
                                    pubinfo = []
                            
                            if ('references' in dataframe_list):
                                pub2refdf = pd.DataFrame(pub2ref, columns = ['CitingPublicationId', 'CitedPublicationId'])
                                if preprocess:
                                    fname = os.path.join(self.path2database, self.path2pub2ref, '{}{}.{}'.format(self.path2pub2ref, ifile, self.database_extension))
                                    self.save_data_file(pub2refdf, fname, key =self.path2pub2ref)

                                    pub2ref = []

                            if ('publicationauthoraffiliation' in dataframe_list):
                                paadf = pd.DataFrame(paa, columns = paa_column_names)
                                if preprocess:
                                    fname = os.path.join(self.path2database, self.path2paa, '{}{}.{}'.format(self.path2paa, ifile, self.database_extension))
                                    self.save_data_file(paadf, fname, key =self.path2paa)

                                    paa = []

                            if ('fields' in dataframe_list) or ('concepts' in dataframe_list):
                                pub2fielddf = pd.DataFrame(pub2field, columns = pub2field_column_names)
                                if preprocess:
                                    fname = os.path.join(self.path2database, self.path2pub2field, '{}{}.{}'.format(self.path2pub2field, ifile, self.database_extension))
                                    self.save_data_file(pub2fielddf, fname, key =self.path2pub2field)

                                    pub2field = []

                            if ('abstracts' in dataframe_list):
                                with gzip.open(os.path.join(self.path2database, self.path2pub2abstract, '{}{}.jsonl.gz'.format(self.path2pub2abstract, ifile)), 'wb') as outfile:
                                    for abentry in pub2abstract:
                                        outfile.write((json.dumps({abentry[0]:abentry[1]}) + "\n").encode('utf-8'))

                                pub2abstract = []

                            ifile += 1


                
                if ('publications' in dataframe_list) or ('works' in dataframe_list):
                    pub = pd.DataFrame(pubinfo, columns = pub_column_names)
                    if preprocess:
                        fname = os.path.join(self.path2database, self.path2pub, '{}{}.{}'.format(self.path2pub, ifile, self.database_extension))
                        self.save_data_file(pub, fname, key =self.path2pub)
                        
                        pubinfo = []
                
                if ('references' in dataframe_list):
                    pub2refdf = pd.DataFrame(pub2ref, columns = ['CitingPublicationId', 'CitedPublicationId'])
                    if preprocess:
                        fname = os.path.join(self.path2database, self.path2pub2ref, '{}{}.{}'.format(self.path2pub2ref, ifile, self.database_extension))
                        self.save_data_file(pub2refdf, fname, key =self.path2pub2ref)

                        pub2ref = []

                if ('publicationauthoraffiliation' in dataframe_list):
                    paadf = pd.DataFrame(paa, columns = paa_column_names)
                    if preprocess:
                        fname = os.path.join(self.path2database, self.path2paa, '{}{}.{}'.format(self.path2paa, ifile, self.database_extension))
                        self.save_data_file(paadf, fname, key =self.path2paa)

                        paa = []

                if ('fields' in dataframe_list) or ('concepts' in dataframe_list):
                    pub2fielddf = pd.DataFrame(pub2field, columns = pub2field_column_names)
                    if preprocess:
                        fname = os.path.join(self.path2database, self.path2pub2field, '{}{}.{}'.format(self.path2pub2field, ifile, self.database_extension))
                        self.save_data_file(pub2fielddf, fname, key =self.path2pub2field)

                        pub2field = []

                if ('abstracts' in dataframe_list):
                    with gzip.open(os.path.join(self.path2database, self.path2pub2abstract, '{}{}.gz'.format(self.path2pub2abstract, ifile)), 'w') as outfile:
                        for abentry in pub2abstract:
                            outfile.write((json.dumps({abentry[0]:abentry[1]}) + "\n").encode('utf-8'))

                    pub2abstract = []

                ifile += 1

        if preprocess_dicts and (('publications' in dataframe_list) or ('works' in dataframe_list)) and preprocess_dicts:
            with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
                outfile.write(json.dumps(pub2year).encode('utf8'))

            with gzip.open(os.path.join(self.path2database, 'pub2doctype.json.gz'), 'w') as outfile:
                outfile.write(json.dumps(pub2doctype).encode('utf8'))

        return True

    def parse_concepts(self, preprocess = True, specific_update='', show_progress=True):
        """
        Parse the MAG Paper Field raw data.

        Parameters
        ----------
        preprocess: bool, default True
            Save the processed data in new DataFrames.

        specific_update: str
            Parse only a specific update date, specified by the date in Y-M-D format, 2022-01-01.  
            If empty the full data is parsed.

        show_progress: bool, default True
            Show progress with processing of the data.

        Returns
        ----------
        DataFrame
            FieldInfo DataFrame.
        """

        if specific_update == '' or specific_update is None:
            concept_dir = os.path.join(self.path2database, 'data', 'concepts')
        else:
            concept_dir = os.path.join(self.path2database, 'data/concepts', 'updated_date={}'.format(specific_update))

        if not os.path.exists(concept_dir):
            raise ValueError("The concepts data was not found in the DataBase path.  Please download the data before parsing.")
        else:
            files_to_parse = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(concept_dir) for file in filenames if '.gz' in file]


        field_column_names = ['FieldId', 'FieldName', 'WikiData', 'FieldLevel', 'NumberPublications', 'NumberCitations']

        if preprocess:
            if not os.path.exists(os.path.join(self.path2database, self.path2fieldinfo )):
                os.mkdir(os.path.join(self.path2database, self.path2fieldinfo ))

        fieldinfo = []
        fieldhierarchy = []
        for file_name in tqdm(files_to_parse, desc='Concepts', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(concept_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        cline = json.loads(line)
                        fid = self.clean_openalex_ids(cline.get('id', None))
                        fielddata = [fid]
                        fielddata += [cline.get(prop, None) for prop in ['display_name', 'wikidata']]
                        fielddata += [load_int(cline.get(prop, None)) for prop in ['level', 'works_count', 'cited_by_count']]
                        
                        fieldinfo.append(fielddata)

                        for an in cline.get('ancestors', []):
                            fieldhierarchy.append([self.clean_openalex_ids(an.get('id', None)), fid])
                
                    

        fieldinfo = pd.DataFrame(fieldinfo, columns = field_column_names)
        if preprocess:
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format(self.path2fieldinfo, 0, self.database_extension))
            self.save_data_file(fieldinfo, fname, key =self.path2fieldinfo)

            fieldhierarchy = pd.DataFrame(fieldhierarchy, columns = ['ParentFieldId', 'ChildFieldId'])
            fname = os.path.join(self.path2database, self.path2fieldinfo, '{}{}.{}'.format('fieldhierarchy', 0, self.database_extension))
            self.save_data_file(fieldhierarchy, fname, key ='fieldhierarchy')

        return fieldinfo


