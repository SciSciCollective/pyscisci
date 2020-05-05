from pyscisci.database import BibDataBase

class WOS(BibDataBase):
    """
    Base class for Web of Science interface.

    TODO: everything

    """

    def __init__(self, path2database = ''):

        self.path2database = path2database