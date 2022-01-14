
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

openalex_works_dfset = {'publications', 'references', 'publicationauthoraffiliation', 'fields', 'abstracts'}

class OpenAlex(BibDataBase):
    """
    Base class for the OpenAlex interface.

    This is an extension of 'BibDataBase' with processing functions specific to the MAG.
    See 'BibDataBase' in database.py for details of non-MAG specific functions.

    The MAG comes structured into three folders: mag, advanced, nlp.
    Explain downloading etc.

    """

    def __init__(self, path2database = '', keep_in_memory = False, global_filter = None, show_progress=True):

        self._default_init(path2database, keep_in_memory, global_filter, show_progress)

        self.PublicationIdType = int
        self.AffiliationIdType = int
        self.AuthorIdType = int
        self.JournalIdType = int

    def preprocess(self, dflist = None, show_progress=True):
        """
        Bulk preprocess the MAG raw data.

        Parameters
        ----------
        dflist: list, default None
            The list of DataFrames to preprocess.  If None, all MAG DataFrames are preprocessed.
            
        show_progress: bool, default True
            Show progress with processing of the data.

        """
        if dflist is None or 'all' in dflist:
            dflist = openalex_works_dfset.union(set(['affiliations', 'authors', 'venues', 'concepts']))

        pubdflist = list(set(dflist).intersection(openalex_works_dfset))

        if 'affiliations' in dflist:
            self.parse_affiliations(preprocess = True, show_progress=show_progress)

        if 'authors' in dflist:
            self.parse_authors(preprocess = True, show_progress=show_progress)

        if 'publications' in dflist:
            self.parse_publications(preprocess = True, dflist=pubdflist, show_progress=show_progress)

        if 'concepts' in dflist:
            self.parse_fields(preprocess=True, show_progress=show_progress)

        if 'venues' in dflist:
            self.parse_venues(preprocess=True, show_progress=show_progress)

    def download_from_source(self, aws_access_key_id='', aws_secret_access_key='', specific_update='',
        dflist = 'all',
        rewrite_existing = False, show_progress=True):

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

        dflist: list
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

        show_progress: bool, default True
            Show progress with processing of the data.
        
        # taken from https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket
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

        
        bucket = 'openalex'

        openalex_data2download = []
        if 'all' in dflist:
            openalex_data2download = ['authors', 'institutions', 'venues', 'works', 'concepts']
        else:
            if 'affiliation' in dflist:
                openalex_data2download.append('institutions')
            if 'author' in dflist:
                openalex_data2download.append('authors')
            if 'publication' in dflist:
                openalex_data2download.append('works')
                openalex_data2download.append('venues')
            if 'reference' in dflist or 'publicationauthoraffiliation' in dflist or 'abstracts' in dflist:
                openalex_data2download.append('works')
            if 'fields' in dflist:
                openalex_data2download.append('works')
                openalex_data2download.append('concepts')

        openalex_data2download = set(openalex_data2download)

        edit_works = not all([df in dflist for df in ['references', 'publicationauthoraffiliation', 'fields', 'abstracts']])


        if not rewrite_existing:
            files_already_downloaded = [os.path.join(dirpath, file) for (dirpath, dirnames, filenames) in os.walk(self.path2database) for file in filenames]
        else:
            files_already_downloaded = []


        keys = []
        dirs = []
        next_token = ''
        
        base_kwargs = {'Bucket':bucket}
        while next_token is not None:
            kwargs = base_kwargs.copy()
            if next_token != '':
                kwargs.update({'ContinuationToken': next_token})
            results = s3_client.list_objects_v2(**kwargs)
            
            for cdict in results.get('Contents', {}):
                k = cdict.get('Key')

                if not k in files_already_downloaded:
                    if 'data' in k: 
                        if k.split('/')[1] in openalex_data2download and (specific_update=='' or k.split('/')[2] == 'updated_date={}'.format(specific_update)):
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
            
            s3_client.download_file(bucket, k, dest_pathname)

            # the works contains references, concepts, and astracts, check if we keep these
            if edit_works and (len(k.split('/')) > 1 and k.split('/')[1] == 'works' and k[-3:]=='.gz'):
                with gzip.open(dest_pathname, 'r') as infile:
                    with gzip.open(dest_pathname.split('.')[0] + '_temp.gz', 'w') as outfile:
                        for line in infile:
                            jline = json.loads(line)

                            if ( not ('fields' in dflist) ) and 'concepts' in jline:
                                del jline['concepts']

                            if ( not ('references' in dflist) ) and 'referenced_works' in jline:
                                del jline['referenced_works']

                            if ( not ('publicationauthoraffiliation' in dflist) ) and 'authorships' in jline:
                                del jline['authorships']

                            if ( not ('abstracts' in dflist) ) and 'abstract_inverted_index' in jline:
                                del jline['abstract_inverted_index']

                            newline = json.dumps(jline) + "\n"
                            outfile.write(newline.encode('utf-8'))

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


        if not os.path.exists(os.path.join(self.path2database, 'affiliation')):
            os.mkdir(os.path.join(self.path2database, 'affiliation'))

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
                aff.to_hdf(os.path.join(self.path2database, 'affiliation', 'affiliation{}.hdf'.format(ifile)),
                                                                            key = 'affiliation', mode = 'w')
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
            if not os.path.exists(os.path.join(self.path2database, 'author')):
                os.mkdir(os.path.join(self.path2database, 'author'))

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
                            hname = HumanName(unicodedata.normalize('NFD', aline.get('display_name', '')))
                            authline += [hname.last, hname.first, hname.middle]

                        authorinfo.append(authline)

                author = pd.DataFrame(authorinfo, columns = author_column_names)

                # time to save
                if preprocess:
                    author.to_hdf(os.path.join(self.path2database, 'author', 'author{}.hdf'.format(ifile)),
                                                                                key = 'author', mode = 'w')
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
            if not os.path.exists(os.path.join(self.path2database, 'venue')):
                os.mkdir(os.path.join(self.path2database, 'venue'))


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
                    venue.to_hdf(os.path.join(self.path2database, 'venue', 'venue{}.hdf'.format(ifile)), key = 'venue', mode = 'w')
                    ifile += 1
                    venue_info = []
        
        return venue


    def parse_publications(self, preprocess = True, specific_update='', preprocess_dicts = True, 
        dflist = ['publications', 'references', 'publicationauthoraffiliation', 'fields'],
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

        dflist: list
            The data types to download and save from OpenAlex.
                'all'
                'publication'
                'reference'
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


        if 'all' in dflist:
            dflist = openalex_works_dfset


        pub_column_names = ['PublicationId', 'JournalId', 'Year', 'NumberCitations', 'Doi', 'Title', 'Date', 'DocType', 'PMID', 'Volume', 'Issue', 'FirstPage', 'LastPage', 'IsRetracted', 'IsParaText']

        if preprocess and ('publications' in dflist):
            if not os.path.exists(os.path.join(self.path2database, 'publication')):
                os.mkdir(os.path.join(self.path2database, 'publication'))

            pubinfo = []
            pub2year = {}
            pub2doctype = {}

        if preprocess and ('references' in dflist):
            if not os.path.exists(os.path.join(self.path2database, 'pub2ref')):
                os.mkdir(os.path.join(self.path2database, 'pub2ref'))

            pub2ref = []

        paa_column_names = ['PublicationId', 'AuthorId', 'AffiliationId', 'AuthorSequence', 'OrigAuthorName']

        if preprocess and ('publicationauthoraffiliation' in dflist):
            if not os.path.exists(os.path.join(self.path2database, 'paa')):
                os.mkdir(os.path.join(self.path2database, 'paa'))

            paa = []

        pub2field_column_names = ['PublicationId', 'FieldId', 'FieldLevel', 'Score']

        if preprocess and ('fields' in dflist):
            if not os.path.exists(os.path.join(self.path2database, 'pub2field')):
                os.mkdir(os.path.join(self.path2database, 'pub2field'))

            if not os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
                os.mkdir(os.path.join(self.path2database, 'fieldinfo'))

            fieldinfo = {}
            pub2field = []

        if preprocess and ('abstracts' in dflist):
            if not os.path.exists(os.path.join(self.path2database, 'pub2abstract')):
                os.mkdir(os.path.join(self.path2database, 'pub2abstract'))

            pub2abstract = []

        ifile = 0
        for file_name in tqdm(files_to_parse, desc='Works', leave=True, disable=not show_progress):
            with gzip.open(os.path.join(work_dir, file_name), 'r') as infile:
                for line in infile:
                    if line != '\n'.encode():
                        wline = json.loads(line)
                        
                        ownid = self.clean_openalex_ids(wline['id'])

                        if ('all' in dflist or 'publication' in dflist):
                            wdata = [ownid, self.clean_openalex_ids(wline.get('host_venue', {}).get('id', None))]
                            wdata += [load_int(wline.get(idname, None)) for idname in ['publication_year', 'cited_by_count']]
                            wdata += [wline.get(idname, None) for idname in ['doi', 'title', 'publication_date', 'type']]
                            wdata += [wline.get('pmid', None)]
                            bibinfo = wline.get('biblio', {})
                            wdata += [bibinfo.get(idname, None) for idname in ['volume', 'issue', 'first_page', 'last_page']]
                            wdata += [load_bool(wline.get(extraid, None)) for extraid in ['is_retracted', 'is_paratext']]

                            if preprocess_dicts:
                                pub2year[wdata[0]] = wdata[2]
                                pub2doctype[wdata[0]] = wdata[7]

                            pubinfo.append(wdata)

                        if ('references' in dflist):
                            pub2ref.extend([[ownid, self.clean_openalex_ids(citedid)] for citedid in wline.get('referenced_works', [])])


                        if ('publicationauthoraffiliation' in dflist):
                            iauthor = 1
                            for authorinfo in wline.get('authorships', []):
                                authorid = self.clean_openalex_ids(authorinfo.get('author', {}).get('id', None))
                                authorname = authorinfo.get('author', {}).get('display_name', '')
                                
                                if (iauthor == 1 and authorinfo.get('author_position', None) != 'first'):
                                    iauthor = None
                                    #print(wline.get('authorships', []))
                                    #raise ValueError('author position')

                                institution_list = wline.get('institutions', [])
                                if len(institution_list) == 0:
                                    institution_list = [None]
                                else:
                                    institution_list = [self.clean_openalex_ids(affinfo.get('id', None)) for affinfo in institution_list]

                                for inid in institution_list:
                                    paa.append([ownid, authorid, inid, iauthor, authorname])

                                if not iauthor is None:
                                    iauthor += 1

                        if ('fields' in dflist):
                            pub_concepts = wline.get('concepts', [])
                            for con_dict in pub_concepts:
                                pub2field.append([ownid, self.clean_openalex_ids(con_dict.get('id', None)), load_int(con_dict.get('level', None)), load_float(con_dict.get('score', None))])
                        

                        if ('abstracts' in dflist) and 'abstract_inverted_index' in wline:
                            pub2abstract.append([ownid, wline['abstract_inverted_index']])


                
                if ('publication' in dflist):
                    pub = pd.DataFrame(pubinfo, columns = pub_column_names)
                    if preprocess:
                        pub.to_hdf(
                            os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)),
                                                                                    key = 'publication', mode = 'w')
                        
                        pubinfo = []
                
                if ('references' in dflist):
                    pub2refdf = pd.DataFrame(pub2ref, columns = ['CitingPublicationId', 'CitedPublicationId'])
                    if preprocess:
                        pub2refdf.to_hdf(os.path.join(self.path2database, 'pub2ref', 'pub2ref{}.hdf'.format(ifile)),
                                                                            key = 'pub2ref', mode = 'w')
                        pub2ref = []

                if ('publicationauthoraffiliation' in dflist):
                    paadf = pd.DataFrame(paa, columns = paa_column_names)
                    if preprocess:
                        paadf.to_hdf(os.path.join(self.path2database, 'publication', 'publication{}.hdf'.format(ifile)),
                                                                                    key = 'publication', mode = 'w')
                        paa = []

                if ('fields' in dflist):
                    pub2fielddf = pd.DataFrame(pub2field, columns = pub2field_column_names)
                    if preprocess:
                        pub2fielddf.to_hdf(os.path.join(self.path2database, 'pub2field', 'pub2field{}.hdf'.format(ifile)),
                                                                            key = 'pub2field', mode = 'w')
                        pub2field = []

                if ('abstracts' in dflist):
                    print(ifile, len(pub2abstract))
                    with gzip.open(os.path.join(self.path2database, 'pub2abstract', 'pub2abstract{}.gzip'.format(ifile)), 'w') as outfile:
                        for abentry in pub2abstract:
                            outfile.write((json.dumps({abentry[0]:abentry[1]}) + "\n").encode('utf-8'))

                ifile += 1

        if ('publications' in dflist) and preprocess_dicts:
            with gzip.open(os.path.join(self.path2database, 'pub2year.json.gz'), 'w') as outfile:
                outfile.write(json.dumps(pub2year).encode('utf8'))

            with gzip.open(os.path.join(self.path2database, 'pub2doctype.json.gz'), 'w') as outfile:
                outfile.write(json.dumps(pub2doctype).encode('utf8'))

        return pub

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
            if not os.path.exists(os.path.join(self.path2database, 'fieldinfo')):
                os.mkdir(os.path.join(self.path2database, 'fieldinfo'))

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
            fieldinfo.to_hdf(os.path.join(self.path2database, 'fieldinfo', 'fieldinfo0.hdf'), key = 'fieldinfo', mode = 'w')
            fieldhierarchy = pd.DataFrame(fieldhierarchy, columns = ['ParentFieldId', 'ChildFieldId'])
            fieldhierarchy.to_hdf(os.path.join(self.path2database, 'fieldinfo', 'fieldhierarchy0.hdf'), key = 'fieldhierarchy', mode = 'w')

        return fieldinfo


