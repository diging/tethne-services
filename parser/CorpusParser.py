import pandas as pd
from tethne import Corpus


columns = ["WOSID", "DATE", "TITLE", "LASTNAME", "FIRSTNAME", "JOURNAL", "EMAILADDRESS",
           "PUBLISHER", "SUBJECT", "WC", "AUTHOR_KEYWORDS", "INSTITUTE", "AUTH_LITERAL", "CO-AUTHORS"]


class CorpusParser:
    """CorpusParser : To parse a Tethne Corpus object and return a DataFrame of Author-Paper instances

    This module is responsible for parsing a `Tethne` corpus object and return a pandas DataFrame with 14 columns.
    The columns are :
    >>> ["WOSID", "DATE", "TITLE", "LASTNAME", "FIRSTNAME", "JOURNAL", "EMAILADDRESS",
    >>>  "PUBLISHER", "SUBJECT", "WC", "AUTHOR_KEYWORDS", "INSTITUTE", "AUTH_LITERAL", "CO-AUTHORS"]

    Each row in the DataFrame object is an Author-Paper instance.
    Additionally, each row is assigned an index. The index is generated using (concatenation of) the following:
        1. WOS ID
        2. Author Last Name
        3. Author First Name

    Example:
        >>> from parser.CorpusParser import CorpusParser
        >>> parser = CorpusParser(tethne_corpus=self.corpus)
        >>> df = parser.parse() # final pandas DataFrame of Author-Paper instances.

    Methods:
        __init__(self, tethne_corpus) : Initialisation method for the class `CorpusParser`

        parse(self) : Returns a pandas DataFrame of Author-Paper instances.
    """

    def __init__(self, tethne_corpus):
        """Initialisation(__init__()) for the class `CorpusParser`

        Args:
            tethne_corpus (`Tethne` corpus object)

        Returns:
            `CorpusParser` class instance : The purpose of this method is to create an instance of CorpusParser

        Raises:
            ValueError: If the input parameter `tethne_corpus` is not an object of class `tethne.Corpus`


        """
        if not isinstance(tethne_corpus, Corpus):
            raise ValueError('The input parameter should be a Tethne Corpus object')
        self.corpus = tethne_corpus
        self.records = []
        self.indices = []
        self.df = None

    def parse(self):
        """Parse method : iterates over each paper in the Corpus object and adds it to the pandas DataFrame

        Returns:
            df : A pandas DataFrame with 14 columns. Each row in the dataFrame is an Author-Paper instance.
        """
        for paper in self.corpus:
            set_of_authors = set(paper.authors_full)
            for author in set_of_authors:
                    current_author = set()
                    current_author.add(author)
                    coauthor_set = set_of_authors.difference(current_author)
                    lastname = author[0]
                    firstname = author[1]
                    index = lastname+firstname+getattr(paper, 'wosid')
                    self.indices.append(index)

                    row = (getattr(paper, 'wosid', ''),
                           str(getattr(paper, 'date', '')),
                           getattr(paper, 'title', ''),
                           lastname,
                           firstname,
                           getattr(paper, 'journal', ''),
                           getattr(paper, 'emailAddress', []),
                           getattr(paper, 'publisher', ''),
                           getattr(paper, 'subject', []),
                           getattr(paper, 'WC', ''),
                           getattr(paper, 'authorKeywords', []),
                           getattr(paper, 'authorAddress', ""),
                           lastname+firstname,
                           list(coauthor_set))

                    self.records.append(row)
        self.df = pd.DataFrame(self.records, columns=columns, index=self.indices)
        return self.df




