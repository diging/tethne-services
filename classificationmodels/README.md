# Serialized models
The classification models after training, testing & Cross-validation are persisted here.
These models can be directly loaded and used for prediction: An example is shown below

```python
    import pickle
    from author.paperinstances import Compare
    
    #Load the persisted classifier
    classifier_path = '../classificationmodels/random_forest.pkl'
    classifier = os.path.join(os.path.dirname(__file__), classifier_path)
    with open(classifier, 'r') as output:
        clf = pickle.loads(output.read())
    
    """
    Create an instance of the Compare class. Here, the 2 paper samples passed in input are 2 rows of the pandas Dataframe.
    This DataFrame can be created using the `CorpusParser.py` class using a `tethne.corpus` object.
    
    
    """
    
    compare_instance = Compare(paper_sample1, paper_sample2)
    compare_instance.create_single_record()
    compare_instance.calculate_scores()
    return clf.predict(compare_instance.scores_df[features])
```