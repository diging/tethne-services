from authors.paperinstances import CorpusParser, classify
from tethne import Corpus
from tethne.readers import wos
from fuzzywuzzy import fuzz
from pprint import pprint
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
    """
    IdentityCluster class uses Machine Learning(RandomForestClassifier) to cluster paper instances belonging to the same
    author

    We first build an initial cluster using the class `InitialCluster`.  In this process, we do not use any
    classification or machine learning approach. Initial Clustering is done to limit the size of comparisons when we
    perform the actual classification. While we build this initial cluster, we group together author_literals which are
    similar and have higher probability of actually belonging to the same cluster.

    Algorithm to build the IdentityCluster:

        STEP 1 : Parse the TETHNE corpus and return a pandas DataFrame of Author-paper instances. Please note that an
                 index is assigned to each Author-Paper instance.The index is generated using (concatenation of)
                 the following:
                    1. WOS ID
                    2. Author Last Name
                    3. Author First Name

        STEP 2 : Use the `IdentityCluster` class to group instances belonging to the same class(Basically, build a
                 dictionary).

        STEP 3 : Return the dictionary with LABEL as keys and a set of pandas DataFrame indexes as values. These indices
                 are the same which are created in the STEP 1 of the algorithm

    Example:
        >>> from authors.cluster import IdentityCluster
        >>> from tethne.readers import wos
        >>> from authors.paperinstances import CorpusParser
        >>> datapath = "/Users/aosingh/tethne-services/tests/data/Albertini_David.txt"
        >>> corpus = wos.read(datapath)

        >>> corpus_parser = CorpusParser(tethne_corpus=corpus)
        >>> df = corpus_parser.parse() # STEP 1 in the algorithm

        >>> identity_cluster_instance = IdentityCluster(corpus=corpus)
        >>> identity_clusters = identity_cluster_instance.build() # STEPS 2 and 3 in the algorithm

    """
    def __init__(self, corpus):
        """Initialisation(__init__()) for the class `IdentityCluster`

        Args:
            corpus (`Tethne` corpus object)

        Returns:
            `IdentityCluster` class instance : The purpose of this method is to create an instance of IdentityCluster

        Raises:
            ValueError: If the input parameter `corpus` is not an object of class `tethne.Corpus`


        """
        if not isinstance(corpus, Corpus):
            raise ValueError("The input object should be a Tethne Corpus object")
        self.corpus = corpus
        self.identity_clusters = {}

    def build(self):
        """
        Args:
            self

        Returns:
            `Dictionary` : A set of clusters, where (for each cluster) the key is the Label and the value is the set of
             indices(of the Author-Paper instances), belonging to the cluster.

        Example:
            Please see the dictionary returned below. This output is for a Tethne-Corpus with Papers belonging mainly to
            the Author Boyer Barbara. As you can see the cluster size is the largest for the key 'BOYERB'. You can access
            all the papers from the pandas DataFrame by using the indices.

            {u'ARNOLDJM': set([u'ARNOLDJMWOS:A1986E918400022',
                               u'ARNOLDJMWOS:A1988N184500004']),
             u'BOYERB': set([u'BOYERBCWOS:000076265300004',
                             u'BOYERBCWOS:000077556600009',
                             u'BOYERBCWOS:000086633600013',
                             u'BOYERBCWOS:000171953200027',
                             u'BOYERBCWOS:000186338800013',
                             u'BOYERBCWOS:A1971I198200009',
                             u'BOYERBCWOS:A1972L780300002',
                             u'BOYERBCWOS:A1981MP70900043',
                             u'BOYERBCWOS:A1982QN98300013',
                             u'BOYERBCWOS:A1983RR86600053',
                             u'BOYERBCWOS:A1984TR20700060',
                             u'BOYERBCWOS:A1984TR20700065',
                             u'BOYERBCWOS:A1985AUC5500038',
                             u'BOYERBCWOS:A1985AUC5500046',
                             u'BOYERBCWOS:A1986A349100017',
                             u'BOYERBCWOS:A1986C019700010',
                             u'BOYERBCWOS:A1986E918400022',
                             u'BOYERBCWOS:A1986E918400023',
                             u'BOYERBCWOS:A1987G340700002',
                             u'BOYERBCWOS:A1988N184500004',
                             u'BOYERBCWOS:A1988R225500053',
                             u'BOYERBCWOS:A1989CH57500002',
                             u'BOYERBCWOS:A1991GV28500052',
                             u'BOYERBCWOS:A1992KC97700038',
                             u'BOYERBCWOS:A1992KC97700042',
                             u'BOYERBCWOS:A1995RP17800035',
                             u'BOYERBCWOS:A1995TA77100017',
                             u'BOYERBCWOS:A1996VQ71700035',
                             u'BOYERBCWOS:A1996VT14600003',
                             u'BOYERBWOS:A1996UQ10700011']),
             u'HENRYJJ': set([u'HENRYJJWOS:000077556600009',
                              u'HENRYJQWOS:000076265300004',
                              u'HENRYJQWOS:000086633600013',
                              u'HENRYJQWOS:A1995TA77100017',
                              u'HENRYJQWOS:A1996VQ71700035',
                              u'HENRYJQWOS:A1996VT14600003']),
             u'HILLSD': set([u'HILLSDWOS:000171953200027', u'HILLSDWOS:000186338800013']),
             u'KAPLANIM': set([u'KAPLANIMWOS:A1988R225500053',
                               u'KAPLANIMWOS:A1992KC97700042']),
             u'LADURNERP': set([u'LADURNERPWOS:A1996UQ10700011']),
             u'LANDOLFAM': set([u'LANDOLFAMAWOS:A1985AUC5500046',
                                u'LANDOLFAMAWOS:A1986E918400022',
                                u'LANDOLFAMWOS:A1988N184500004']),
             u'MAIRG': set([u'MAIRGWOS:A1996UQ10700011']),
             u'MARTINDALEMQ': set([u'MARTINDALEMQWOS:000077556600009',
                                   u'MARTINDALEMQWOS:000086633600013',
                                   u'MARTINDALEMQWOS:A1995TA77100017',
                                   u'MARTINDALEMQWOS:A1996VQ71700035',
                                   u'MARTINDALEMQWOS:A1996VT14600003']),
             u'PALASZEWSKIPP': set([u'PALASZEWSKIPPWOS:A1983RR86600053']),
             u'REITERD': set([u'REITERDWOS:A1996UQ10700011']),
             u'RIEGERR': set([u'RIEGERRWOS:A1996UQ10700011']),
             u'ROONEYLM': set([u'ROONEYLMWOS:A1984TR20700065']),
             u'SALVENMOSERW': set([u'SALVENMOSERWWOS:A1996UQ10700011']),
             u'SANTOSKA': set([u'SANTOSKAWOS:A1988R225500053']),
             u'SMITHGW': set([u'SMITHGWWOS:A1982QN98300013'])}
        """
        parser = CorpusParser(tethne_corpus=self.corpus)
        df = parser.parse()
        initial_cluster_instance = InitialCluster(corpus=self.corpus)
        initial_clusters = initial_cluster_instance.build()
        for x in initial_clusters:
            self.identity_clusters[x] = set()
            current_df = df[df['AUTH_LITERAL'].isin([x])]
            for index, _ in current_df.iterrows():
                self.identity_clusters[x].add(index)

            current_block = df[df['AUTH_LITERAL'].isin(initial_clusters[x])]

            if len(current_block) > 1:
                for index, row in current_block.iterrows():
                    counter = False
                    for index_child, row_child in current_block.iterrows():
                        if index != index_child:
                            score = classify(row, row_child)
                            if score[0] == 1:
                                counter = True
                                self.identity_clusters[x].add(index)
                            if counter:
                                break
        return self.identity_clusters



corpus = wos.read('/Users/aosingh/tethne-services/tests/data/Boyer_Barbara.txt')
idenityobj = IdentityCluster(corpus=corpus)
idenityobj.build()

