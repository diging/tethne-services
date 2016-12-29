import sys
import unittest

import pandas as pd
from tethne.readers import wos
from authors.paperinstances import CorpusParser

sys.path.append('./')
datapath = './data/Albertini_David.txt'
datapath2 ='./data/Boyer_Barbara.txt'


class TestParser(unittest.TestCase):

    def setUp(self):
        self.corpus = wos.read(datapath)
        self.corpus2 = wos.read(datapath2)

    def test_parser(self):
        parser = CorpusParser(tethne_corpus=self.corpus)
        df = parser.parse()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 14)

    def test_indexing(self):
        index = 'BOYERBCWOS:000076265300004'
        parser = CorpusParser(tethne_corpus=self.corpus2)
        df = parser.parse()
        try:
            paper_instance = df.loc[index]
            print paper_instance
            self.assertEqual(paper_instance['WOSID'], 'WOS:000076265300004')
            self.assertEqual(paper_instance['AUTH_LITERAL'], 'BOYERBC')
        except KeyError:
            self.assertTrue(False)

