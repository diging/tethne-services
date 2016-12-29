# Serialized models
The classification models after training, testing & Cross-validation are persisted here.
These models can be directly loaded and used for classification of 2 paper samples.
An example is shown below:

```python
    import pickle
    from tethne.readers import wos
    from authors.paperinstances import CorpusParser, classify
 
    
    #define the datapath 
    datapath = '/Users/aosingh/tethne-services/tests/data/Boyer_Barbara.txt'
    
    # Parse and get the pandas DataFrame
    corpus= wos.read(datapath)
    parser = CorpusParser(tethne_corpus=corpus)
    df = parser.parse()
    
       
    # Define the indices for 2 paper samples that you want to compare.
    # (Please note that these indices are returned as part of the final identity clusters)
    
    # I have defined them for the purpose of explanation and to show how the corresponding paper-instance can be accessed and classified.
    
    index1 = 'BOYERBCWOS:000076265300004'
    index2 = 'BOYERBWOS:A1996UQ10700011'
    
    # Call the classify method for the 2 paper instances. This method internally loads the 
    # persisted RandomForest classifier.
    
    match = classify(self.df.loc[index1], self.df.loc[index2])
    
    # if match == 1 then, papers belong to the same author, else they do not.
    if match == [1]:
        print "Papers belong to the same Author"
    else:
        print "Papers do not belong to the same Author"
```