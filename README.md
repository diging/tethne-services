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

* `./tethne-services/parser/CorpusParser.py`: 
This module is responsible for parsing a `Tethne` corpus object and returns a pandas DataFrame with 14 columns. 
Each row in the DataFrame is an Author-Papers instance.

    * An example usage of this API is shown below.
    
        ```python
        from parser.CorpusParser import CorpusParser
        from tethne.readers import wos
        datapath = './data/Albertini_David.txt'
        corpus = wos.read(datapath)
        parser = CorpusParser(tethne_corpus=corpus)
        df = parser.parse() # returns pandas DataFrame of Author-Paper instances.
        
        #The DataFrame returned has the following colmns
        ["WOSID", "DATE", "TITLE", "LASTNAME", "FIRSTNAME", "JOURNAL", "EMAILADDRESS",
        "PUBLISHER", "SUBJECT", "WC", "AUTHOR_KEYWORDS", "INSTITUTE", "AUTH_LITERAL", "CO-AUTHORS"]
        
        ```
     

* `./tethne-services/authors/Cluster.py (InitialCluster)`: 
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
        initial_cluster = InitialCluster(corpus=corpus)
        clusters = initial_cluster.build()
        ```



 

