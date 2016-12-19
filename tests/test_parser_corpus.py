from tethne.readers import wos
from parser.CorpusParser import CorpusParser
import unittest
import pandas as pd
import sys

sys.path.append('./')
datapath = './data/Albertini_David.txt'


class TestParser(unittest.TestCase):

    def setUp(self):
        self.corpus = wos.read(datapath)

    def test_parser(self):
        parser = CorpusParser(tethne_corpus=self.corpus)
        df = parser.parse()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 14)
