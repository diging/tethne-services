import os
print os.getcwd()

import unittest

from tethne.readers import wos
from authors.cluster import InitialCluster
datapath = './data/Albertini_David.txt'


class TestInitialCluster(unittest.TestCase):

    def setUp(self):
        self.corpus = wos.read(datapath)
        self.label = 'ALBERTINDF'
        self.members = [u'ALBERTINDF',
                     u'ALBERTINID',
                     u'ALBERTINID F',
                     u'ALBERTINIDAVID',
                     u'ALBERTINIDAVID F',
                     u'ALBERTINIDF']

    def test_initial_clusters(self):
        initial_cluster = InitialCluster(corpus=self.corpus)
        initial_clusters = initial_cluster.build()

        members = initial_clusters[self.label]
        self.assertSetEqual(members, set(self.members))

        for x in self.members:
            if x != self.label and x in initial_clusters:
                self.assertTrue(False, "Members cannot be Cluster Labels")



