from authors.paperinstances import CorpusParser
from tethne import Corpus
from tethne.readers import wos
from fuzzywuzzy import fuzz
import numpy as np
import logging


logger = logging.getLogger('AuthorCluster')


class InitialCluster:
    """InitialCluster, as the name suggests, groups Author-Paper instances by similar Author names. In this process, we
    do not use any classification or machine learning approach. Initial Clustering is done to limit the size of
    comparisons when we perform the actual classification.
    For example: It is not efficient to compare papers by the authors 'BRUCE WAYNE' and 'CLARK KENT' using
    the classification model. We know they are 2 different people.

    While we build this initial cluster, we group together author_literals which are similar and have higher probability
    of actually belonging to the same cluster.

    Example:
        >>> from authors.cluster import InitialCluster
        >>> from tethne.readers import wos
        >>> datapath = './data/Albertini_David.txt'
        >>> corpus = wos.read(datapath)
        >>> initial_cluster = InitialCluster(corpus=corpus)
        >>> clusters = initial_cluster.build()


    """
    def __init__(self, corpus):
        """Initialisation(__init__()) for the class `InitialCluster`

        Args:
            corpus (`Tethne` corpus object)

        Returns:
            `InitialCluster` class instance : The purpose of this method is to create an instance of InitialCluster

        Raises:
            ValueError : If the input parameter `corpus` is not an object of class `tethne.Corpus`
        """
        if not isinstance(corpus, Corpus):
            raise ValueError('The input parameter should be a Tethne Corpus object')
        self.corpus = corpus
        self.initial_clusters = {}

    def build(self):
        """ This method builds the initial cluster from the `tethne.Corpus` object.

        Returns:
            initial_clusters (Dict) : A python dictionary objects where Labels are keys and the set of
            members(similar author literals) are the values corresponding to each key.

        Example:
        >>> {u'AALBERGJ': {u'AALBERGJ', u'VALBERGPA'},
        >>>  u'ABOULGHARM': {u'ABOULGHARM'},
        >>>    u'ABSEDANNIE': {u'ABSEDANNIE'},
        >>>   u'AHUJAKAMAL': {u'AHUJAKAMAL'},
        >>>    u'AINSLIEA': {u'AINSLIEA']},
        >>>    u'AKKOYUNLUGOKHAN': {u'AKKOYUNLUGOKHAN'},
        >>>    u'ALBERTINDF': {u'ALBERTINDF',
        >>>             u'ALBERTINID',
        >>>             u'ALBERTINID F',
        >>>             u'ALBERTINIDAVID',
        >>>             u'ALBERTINIDAVID F',
        >>>             u'ALBERTINIDF'},
        >>>    u'ALECCIC':{u'ALECCIC'},
        >>>    u'ALEXANDREHENRI': {u'ALEXANDREHENRI'},
        >>>    u'ALIKANIMINA': {u'ALIKANIMINA', u'GALIANIDALIA'},
        >>>    u'ALLWORTHAE': {u'ALLWORTHAE'},
        >>>    u'ANDERSENCY': {u'ANDERSENCY', u'ANDERSONE', u'ANDERSONR'}}


        :return:
        """
        parser = CorpusParser(tethne_corpus=self.corpus)
        df = parser.parse()
        unclassified = np.sort(np.array(df.AUTH_LITERAL.unique()))
        assigned = set()
        for x in unclassified:
            match = False
            if x not in assigned:
                for k in self.initial_clusters.keys():
                    if x == k or max(fuzz.ratio(x, k), fuzz.ratio(k, x)) >= 70:
                        self.initial_clusters[k].add(x)
                        assigned.add(x)
                        match = True
                        break
                if not match:
                    self.initial_clusters[x] = set()
                    self.initial_clusters[x].add(x)
                    assigned.add(x)
        logger.debug("Size of the initial Cluster is %s", len(self.initial_clusters))
        return self.initial_clusters


class IdentityCluster:
    def __init__(self, corpus):
        if not isinstance(corpus, Corpus):
            raise ValueError("The input object should be a Tethne Corpus object")
        self.corpus = corpus
        self.identity_clusters = {}

    def build(self):
        parser = CorpusParser(tethne_corpus=self.corpus)
        df = parser.parse()
        initial_cluster_instance = InitialCluster(corpus=self.corpus)
        initial_clusters = initial_cluster_instance.build()
        for x in initial_clusters:
            current_block = df[df['AUTH_LITERAL'].isin(initial_clusters[x])]

            if len(current_block) > 1:
                for index, row in current_block.iterrows():
                    for index_child, row_child in current_block.iterrows():
                        d = {}





corpus = wos.read('/Users/aosingh/tethne-services/tests/data/Albertini_David.txt')
idenityobj = IdentityCluster(corpus=corpus)
idenityobj.build()

