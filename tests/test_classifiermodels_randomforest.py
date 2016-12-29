import unittest
import pickle

import pandas as pd
from tethne.readers import wos
from authors.paperinstances import CorpusParser
from authors.paperinstances import classify

datapath = './data/Boyer_Barbara.txt'


class TestRandomForestClassifier(unittest.TestCase):

    def setUp(self):
        corpus= wos.read(datapath)
        parser = CorpusParser(tethne_corpus=corpus)
        self.df = parser.parse()

    def test_positive_classification(self):
        index1 = 'BOYERBCWOS:000076265300004'
        index2 = 'BOYERBWOS:A1996UQ10700011'
        match = classify(self.df.loc[index1], self.df.loc[index2])
        self.assertEqual(match, [1])

    def test_negative_classification(self):
        index1 = 'BOYERBCWOS:000076265300004'
        index2 = 'MARTINDALEMQWOS:000077556600009'
        match = classify(self.df.loc[index1], self.df.loc[index2])
        self.assertEqual(match, [0])
