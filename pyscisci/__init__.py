__package__ = 'pyscisci'
__title__ = 'pyscisci: A python package for the science of science'
__description__ = 'Lets study science!'

__copyright__ = '2020, Gates, A.J.'

__author__ = """\n""".join([
    'Alexander J Gates <ajgates42@gmail.com>'
])

__version__ = '0.0.1'
__release__ = '0.0.1'

from .metrics import *
from .author import Author
from .publication import Publication
from .bibdatabase import BibDatabase
