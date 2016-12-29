# tethne-services
Tools to enhance metadata-based analysis in Tethne.


### Clone this Repo
Clone this repository into whatever directory you'd like to work on it from:

```bash
git clone https://github.com/diging/tethne-services.git
```

### Install the following
*   [Python v2.7.12](https://www.python.org/downloads/release/python-2712/)
*   [Tethne](http://pythonhosted.org/tethne/)
    *   `pip install tethne`
*   [pandas v0.19.0](http://pandas.pydata.org/)
    *   `pip install pandas`
*   [scikit-learn v0.18.1](http://scikit-learn.org/stable/)
    *   `pip install -U scikit-learn`
*   [numpy v1.11.3](http://www.numpy.org/)
    *   `pip install numpy`
*   [fuzzywuzzy v0.14.0](https://pypi.python.org/pypi/fuzzywuzzy)
    *   `pip install fuzzywuzzy`


### APIs

This module will expose the following APIs:

* `CorpusParser.py`: 
This module is responsible for parsing a `Tethne` corpus object and returns a pandas DataFrame with 14 columns. 
Each row in the DataFrame is an Author-Paper instance. Also, please note that each row in the DataFrame is assigned an index.
The index is generated using (concatenation of)the following:
                    1. WOS ID         
                    2. Author Last Name             
                    3. Author First Name

    * An example usage of this API is shown below.
    
        ```python
        from authors.paperinstances import CorpusParser
        from tethne.readers import wos
        datapath = "/Users/aosingh/tethne-services/tests/data/Albertini_David.txt"
        corpus = wos.read(datapath)
        corpus_parser = CorpusParser(tethne_corpus=corpus)
        df = corpus_parser.parse() # final pandas DataFrame of Author-Paper instances.
        
        #This is how the indices look like for each row
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
        
        #The DataFrame returned has the following colmns
        ["WOSID", "DATE", "TITLE", "LASTNAME", "FIRSTNAME", "JOURNAL", "EMAILADDRESS",
        "PUBLISHER", "SUBJECT", "WC", "AUTHOR_KEYWORDS", "INSTITUTE", "AUTH_LITERAL", "CO-AUTHORS"]
        
        ```
     
* `InitialCluster.py`: 
InitialCluster, as the name suggests, groups Author-Paper instances by similar author names. In this process, we
do not use any classification or machine learning approach. Initial Clustering is done to limit the size of comparisons when we perform the actual classification.
For example: It is not efficient to compare papers by the authors 'BRUCE WAYNE' and 'CLARK KENT' using the classification model. We know they are 2 different people.
While we build this initial cluster, we group together author_literals which are similar and have higher probability of actually belonging to the same cluster.

    * An example usage of this API is shown below 
    
        ```python
        from authors.Cluster import InitialCluster
        from tethne.readers import wos
        datapath = './data/Albertini_David.txt'
        corpus = wos.read(datapath)
        initial_cluster_instance = InitialCluster(corpus=corpus)
        clusters = initial_cluster_instance.build()
        
        #This is how the dictionary of initial clusters will look like:
        
         {u'AALBERGJ': {u'AALBERGJ', u'VALBERGPA'},
          u'ABOULGHARM': {u'ABOULGHARM'},
          u'ABSEDANNIE': {u'ABSEDANNIE'},
          u'AHUJAKAMAL': {u'AHUJAKAMAL'},
          u'AINSLIEA': {u'AINSLIEA']},
          u'AKKOYUNLUGOKHAN': {u'AKKOYUNLUGOKHAN'},
          u'ALBERTINDF': {u'ALBERTINDF',
                     u'ALBERTINID',
                     u'ALBERTINID F',
                     u'ALBERTINIDAVID',
                     u'ALBERTINIDAVID F',
                     u'ALBERTINIDF'},
           u'ALECCIC':{u'ALECCIC'},
           u'ALEXANDREHENRI': {u'ALEXANDREHENRI'},
           u'ALIKANIMINA': {u'ALIKANIMINA', u'GALIANIDALIA'},
           u'ALLWORTHAE': {u'ALLWORTHAE'},
           u'ANDERSENCY': {u'ANDERSENCY', u'ANDERSONE', u'ANDERSONR'}}
        
        
        ```

* `IdentityCluster.py`: 
IdentityCluster class uses Machine Learning(RandomForestClassifier) to cluster paper instances belonging to the same
author. We first build an initial cluster using the class `InitialCluster`. Please read the example below to understand it's usage

    * An example usage of this API is shown below 
    
        ```python
        """
        Algorithm: 
        
        STEP 1 : Parse the TETHNE corpus and return a pandas DataFrame of Author-paper instances. 
                 Please note that, an index is assigned to each Author-Paper instance.
                 The index is generated using (concatenation of)the following:
                    1. WOS ID
                    2. Author Last Name
                    3. Author First Name

        STEP 2 : Use the `IdentityCluster` class to group instances belonging to the same class(Basically, 
                 build a dictionary).   
              
        STEP 3 : Return the dictionary with LABEL as keys and a set of pandas DataFrame indexes as values. 
        These indices are the same which are created in the STEP 1 of the algorithm
        """
        
         from authors.cluster import IdentityCluster
         from tethne.readers import wos
         from authors.paperinstances import CorpusParser
         datapath = "/Users/aosingh/tethne-services/tests/data/Albertini_David.txt"
         corpus = wos.read(datapath)

         corpus_parser = CorpusParser(tethne_corpus=corpus)
         df = corpus_parser.parse() # STEP 1 in the algorithm

         identity_cluster_instance = IdentityCluster(corpus=corpus)
         identity_clusters = identity_cluster_instance.build() # STEPS 2 and 3 in the algorithm
        
        #This is how the dictionary of final identity clusters will look like:
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
        ```


 

