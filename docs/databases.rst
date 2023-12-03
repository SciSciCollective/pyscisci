The DataBase
======================


*pySciSci* provides a standardized interface for working with several of the major datasets in the Science of Science, including: 
	- `OpenAlex <https://openalex.org/>`_
	- `Microsoft Academic Graph <https://docs.microsoft.com/en-us/academic-services/graph/>`_ (MAG)
	- `Clarivate Web of Science <https://clarivate.com/webofsciencegroup/solutions/web-of-science/>`_ (WOS)
	- `American Physics Society <https://journals.aps.org/datasets>`_ (APS)
	- `PubMed <https://www.nlm.nih.gov/databases/download/pubmed_medline.html>`_
	- `DBLP Computer Science Bibliography <https://dblp.uni-trier.de>`_ (DBLP)

The storage and processing frameworks are highly generalizable, and can be extended to other databases not mentioned here.  Please contribute an interface to your data on the github `project page <https://github.com/SciSciCollective/pyscisci>`_! 

Each dataset is accessed as a customized variant of the *BibDataBase* class, a container of python Pandas data frames `pandas <https://pandas.pydata.org/>`_, that handles all data loading and pre-processing.


Currently, we provide direct data access only to DBLP and PubMed.  All other data must be manually downloaded from the data provider before processing.

To facilitate data movement and lower memory overhead when the complete tables are not required, we pre-process the data tables into smaller chunks.  When loading a table into memory, the user can quickly load the full table by referencing the table name as a database property or specify multiple filters to load only a subset of the data.


Basic Data WorkFlow
-------------------
Every dataset in *pySciSci* is first pre-processed into a standardized tabular format based around DataFrame objects.

First usage only:
	- Download Data
	- Preprocess Data

All other usages:
	- Apply Data Filter
	- Load Only the DataFrame you need


DataSet Examples
-----------------
- `Getting Started With OpenAlex <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting_Started/Getting%20Started%20with%20OpenAlex.ipynb>`_
- `Getting Started With MAG <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20MAG.ipynb>`_
- `Getting Started With WoS <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20WOS.ipynb>`_
- `Getting Started With DBLP <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20DBLP.ipynb>`_
- `Getting Started With APS <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20APS.ipynb>`_
- `Getting Started With PubMed <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20PubMed.ipynb>`_
- `Getting Started With Custom DB <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting_Started/Getting%20Started%20with%20a%20Custom%20DB.ipynb>`_

DataFrames
-----------
Each dataset is partitioned into several DataFrames containing information for different bibliometric objects that are accessed as properties of the BibDataBase:
	- *pub*: The DataFrame keeping publication information, including publication date, journal, title, etc.. Each PubId occurs only once.  Columns depend on the specific datasource. 
	- *author*: The DataFrame keeping author names and personal information. Each AuthorId occurs only once.  Columns depend on the specific datasource. 
	- *affiliation*: The DataFrame keeping affiliations names and websites. Each AffiliationId occurs only once.  Columns depend on the specific datasource. 
	- *journal*: The DataFrame keeping journal names and websites. Each JournalId occurs only once.  Columns depend on the specific datasource. 
	- *fieldinfo*: The DataFrame keeping field names and levels. Each FieldId occurs only once.  Columns depend on the specific datasource. 

There are also DataFrames which contain edge lists linking the different bibliometric data objects:
	- *pub2ref*: The DataFrame linking publications to their references (or citations).
	- *pub2field*: The DataFrame linking publications to their fields.
	- *paa*: The DataFrame linking publications to authors to affiliations.  Columns depend on the specific datasource. 

And two processed DataFrames are created which contain the most popular citation counts for future reference:
	- *impact*: The DataFrame linking publications to their citations counts.  Columns depend on the specific datasource and processing options. 
	- *pub2refnoself*: The DataFrame linking publications to their references where all self-citations are removed.
 

Filters
--------
Some datasets contain a wide-range of publication types, from many times, in many fields, spanning many different topics.  Often, it is useful to focus only on a subset of the available data.  For example, the MAG contains many different document types including journal publications, books, patents, and others.

Filters can be applied to the BibDataBase to ensure only a desired subset of the data is loaded into memory.

There are four default filters provided by *pySciSci*:
	- YearFilter
	- DocTypeFilter
	- FieldFilter
	- JournalFilter 

Property Dictionaries
----------------------
Two property dictionaries are created for quick reference without the need to load the complete publication dataframe:
	- *pub2year*: mapping between the PubId and PubYear 
	- *pub2doctype*: mapping between the PubId and DocType (when available)


BibDataBase
------------
.. automodule:: pyscisci.database
   :members:

OpenAlex
------------------------------------

For initial example, see `Getting Started With OpenAlex <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting_Started/Getting%20Started%20with%20OpenAlex.ipynb>`_.

.. automodule:: pyscisci.datasource.OpenAlex
   :members:

Microsoft Academic Graph (MAG)
------------------------------------

For initial example, see `Getting Started With MAG <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20MAG.ipynb>`_.

.. automodule:: pyscisci.datasource.MAG
   :members:

Web of Science (WoS)
------------------------

For initial example, see `Getting Started With WoS <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20WOS.ipynb>`_.

.. automodule:: pyscisci.datasource.WOS
   :members:

DBLP Computer Science Bibliography (DBLP)
------------------------------------------------

For initial example, see `Getting Started With DBLP <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20DBLP.ipynb>`_.

.. automodule:: pyscisci.datasource.DBLP
   :members:

American Physics Society (APS)
------------------------------------

For initial example, see `Getting Started With APS <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20APS.ipynb>`_.

.. automodule:: pyscisci.datasource.APS
   :members:

PubMed
------------

For initial example, see `Getting Started With PubMed <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting%20Started%20with%20PubMed.ipynb>`_.


.. automodule:: pyscisci.datasource.PubMed
   :members:

Custom DB
------------

For initial example, see `Getting Started With Custom DB <https://github.com/SciSciCollective/pyscisci/blob/master/examples/Getting_Started/Getting%20Started%20with%20a%20Custom%20DB.ipynb>`_.


.. automodule:: pyscisci.datasource.CustomDB
   :members: