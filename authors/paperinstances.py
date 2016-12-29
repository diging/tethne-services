from __future__ import division
import pandas as pd
import re
import os
import pickle

from tethne import Corpus
from ast import literal_eval
from utilities import cosine_similarity, sentence_to_vector
from fuzzywuzzy import fuzz


# LOAD THE SERIALIZED RANDOM FOREST CLASSIFIER
classifier_path = '../classificationmodels/random_forest.pkl'
classifier = os.path.join(os.path.dirname(__file__), classifier_path)
with open(classifier, 'r') as output:
    clf = pickle.loads(output.read())


# COLUMNS returned in the Author-paper instances after parsing a `Tethne` Corpus object
columns = ["WOSID", "DATE", "TITLE", "LASTNAME", "FIRSTNAME", "JOURNAL", "EMAILADDRESS",
           "PUBLISHER", "SUBJECT", "WC", "AUTHOR_KEYWORDS", "INSTITUTE", "AUTH_LITERAL", "CO-AUTHORS"]


# LAMBDA functions to be used when we compare institute names of 2 Author-paper instances.
split_institute = lambda institute:institute.split(',')
join_institute_names = lambda institute : " ".join(x for x in institute)

# Features we will calculate before classification.
features = ['INSTIT_SCORE',
        'BOTH_NAME_SCORE',
        'FNAME_SCORE',
        'FNAME_PARTIAL_SCORE',
        'LNAME_SCORE',
        'LNAME_PARTIAL_SCORE',
        'EMAIL_ADDR_SCORE',
        'AUTH_KW_SCORE',
        'COAUTHOR_SCORE']


class CorpusParser:
    """CorpusParser : To parse a Tethne Corpus object and return a DataFrame of Author-Paper instances

    This module is responsible for parsing a `Tethne` corpus object and return a pandas DataFrame with 14 columns.
    The columns are :
    >>> ["WOSID", "DATE", "TITLE", "LASTNAME", "FIRSTNAME", "JOURNAL", "EMAILADDRESS",
    >>>  "PUBLISHER", "SUBJECT", "WC", "AUTHOR_KEYWORDS", "INSTITUTE", "AUTH_LITERAL", "CO-AUTHORS"]

    Each row in the DataFrame object is an Author-Paper instance.
    Additionally, each row is assigned an index. The index is generated using (concatenation of) the following:
        1. WOS ID
        2. Author Last Name
        3. Author First Name

    Example:
        >>> from authors.paperinstances import CorpusParser
        >>> from tethne.readers import wos
        >>> datapath = "/Users/aosingh/tethne-services/tests/data/Albertini_David.txt"
        >>> corpus = wos.read(datapath)
        >>> corpus_parser = CorpusParser(tethne_corpus=corpus)
        >>> df = corpus_parser.parse() # final pandas DataFrame of Author-Paper instances.

    Methods:
        __init__(self, tethne_corpus) : Initialisation method for the class `CorpusParser`

        parse(self) : Returns a pandas DataFrame of Author-Paper instances.
    """

    def __init__(self, tethne_corpus):
        """Initialisation(__init__()) for the class `CorpusParser`

        Args:
            tethne_corpus (`Tethne` corpus object)

        Returns:
            `CorpusParser` class instance : The purpose of this method is to create an instance of CorpusParser

        Raises:
            ValueError: If the input parameter `tethne_corpus` is not an object of class `tethne.Corpus`


        """
        if not isinstance(tethne_corpus, Corpus):
            raise ValueError('The input parameter should be a Tethne Corpus object')
        self.corpus = tethne_corpus
        self.records = []
        self.indices = []
        self.df = None

    def parse(self):
        """Parse method : iterates over each paper in the Corpus object and adds it to the pandas DataFrame

        Returns:
            df : A pandas DataFrame with 14 columns. Each row in the dataFrame is an Author-Paper instance.
        """
        for paper in self.corpus:
            set_of_authors = set(paper.authors_full)
            for author in set_of_authors:
                    current_author = set()
                    current_author.add(author)
                    coauthor_set = set_of_authors.difference(current_author)
                    lastname = author[0]
                    firstname = author[1]
                    index = lastname+firstname+getattr(paper, 'wosid')
                    self.indices.append(index)

                    row = (getattr(paper, 'wosid', ''),
                           str(getattr(paper, 'date', '')),
                           getattr(paper, 'title', ''),
                           lastname,
                           firstname,
                           getattr(paper, 'journal', ''),
                           getattr(paper, 'emailAddress', []),
                           getattr(paper, 'publisher', ''),
                           getattr(paper, 'subject', []),
                           getattr(paper, 'WC', ''),
                           getattr(paper, 'authorKeywords', []),
                           getattr(paper, 'authorAddress', ""),
                           lastname+firstname,
                           list(coauthor_set))

                    self.records.append(row)
        self.df = pd.DataFrame(self.records, columns=columns, index=self.indices)
        return self.df


class Compare:

    def __init__(self, paper_sample1, paper_sample2):
        self.paper_sample1 = paper_sample1
        self.paper_sample2 = paper_sample2
        self.record_df = None
        self.scores_df = None

    @staticmethod
    def set_feature_value(paper_sample, record, attribute_name, column_name):
        """

        ``Example`` The following example explains the usage.
        >>> Compare.set_feature_value(row, d, 'FIRST_NAME1', 'FIRSTNAME')
            This means, do the following
        >>> d['FIRST_NAME1'] = row['FIRSTNAME']

        :param paper_sample:
        :param record:
        :param attribute_name:
        :param column_name:
        :return: training_record after setting the feature value.
        """

        record[attribute_name] = paper_sample[column_name]
        return record


    def create_single_record(self):
        record = []
        d = {}
        d = Compare.set_feature_value(self.paper_sample1, d, 'FIRST_NAME1', 'FIRSTNAME')
        d = Compare.set_feature_value(self.paper_sample1, d, 'LAST_NAME1', 'LASTNAME')
        d = Compare.set_feature_value(self.paper_sample1, d, 'EMAILADDRESS1', 'EMAILADDRESS')
        d = Compare.set_feature_value(self.paper_sample1, d, 'INSTITUTE1', 'INSTITUTE')
        d = Compare.set_feature_value(self.paper_sample1, d, 'AUTHOR_KW1', 'AUTHOR_KEYWORDS')
        d = Compare.set_feature_value(self.paper_sample1, d, 'COAUTHORS1', 'CO-AUTHORS')

        d = Compare.set_feature_value(self.paper_sample2, d, 'FIRST_NAME2', 'FIRSTNAME')
        d = Compare.set_feature_value(self.paper_sample2, d, 'LAST_NAME2', 'LASTNAME')
        d = Compare.set_feature_value(self.paper_sample2, d, 'EMAILADDRESS2', 'EMAILADDRESS')
        d = Compare.set_feature_value(self.paper_sample2, d, 'INSTITUTE2', 'INSTITUTE')
        d = Compare.set_feature_value(self.paper_sample2, d, 'AUTHOR_KW2', 'AUTHOR_KEYWORDS')
        d = Compare.set_feature_value(self.paper_sample2, d, 'COAUTHORS2', 'CO-AUTHORS')
        record.append(d)
        self.record_df = pd.DataFrame(record)

    @staticmethod
    def get_score_for_coauthors(row):

        """
        Given a training_record, calculate the overlap score in between Co-authors
        The overlap score is calculated in between the fields COAUTHORS1, COAUTHORS2

                >>>intersection = COAUTHORS1 & COAUTHORS2
                >>>union = COAUTHORS1 | COAUTHORS2
                >>>score = len(intersection)/len(union)

        Please read :https://diging.atlassian.net/wiki/display/DILS/Co-authors+for+disambiguation for more details

        :param training_record:
        :return: A score between 0 and 1
        """
        coauthor1 = row['COAUTHORS1']
        coauthor2 = row['COAUTHORS2']
        try:
            if coauthor1 is None or len(coauthor1) == 0:
                return 0
            if coauthor2 is None or len(coauthor2) == 0:
                return 0
            coauthor2 = set(coauthor2)
            coauthor1 = set(coauthor1)

            kw_intersection = coauthor1 & coauthor2
            kw_union = coauthor1 | coauthor2
            if len(kw_union) > 0:
                return len(kw_intersection)/len(kw_union)
            return 0
        except Exception:
            return 0

    @staticmethod
    def get_score_for_author_keywords(row):
        """
        Given a training_record, calculate the overlap score in between Author Keywords
        The overlap score is calculated in between the fields AUTHOR_KW1, AUTHOR_KW2

                >>>intersection = AUTHOR_KW1 & AUTHOR_KW2
                >>>union = AUTHOR_KW1 | AUTHOR_KW2
                >>>score = len(intersection)/len(union)

        Please read : https://diging.atlassian.net/wiki/display/DILS/Author+keywords+for+dismbiguation for more details

        :param training_record:
        :return: A score between 0 and 1
        """
        author_kw1 = row['AUTHOR_KW1']
        author_kw2 = row['AUTHOR_KW2']
        try:
            if author_kw1 is None or len(author_kw1) == 0:
                return 0
            if author_kw2 is None or len(author_kw2) == 0:
                return 0
            author_kw2 = set(author_kw2)
            author_kw1 = set(author_kw1)

            kw_intersection = author_kw1 & author_kw2
            kw_union = author_kw1 | author_kw2
            if len(kw_union) > 0:
                return len(kw_intersection)/len(kw_union)
            return 0
        except Exception:
            "From exception AUTHOR KEYWORDS"
            return 0


    @staticmethod
    def get_score_for_email_address(row):
        """
        Return 1 if both the email-addresses match else return 0.

        Please read : https://diging.atlassian.net/wiki/pages/viewpage.action?pageId=46432257 for more details

        :param training_record:
        :return: A score of either 0 or 1
        """

        email1 = row['EMAILADDRESS1']
        email2 = row['EMAILADDRESS2']
        if email1 is None or len(email1) == 0:
            return 0
        if email2 is None or len(email2) == 0:
            return 0
        if isinstance(email1, list) & isinstance(email2, list):
            list1 = set(email1)
            list2 = set(email2)
            intersection = list1 & list2
            union = list1 | list2
            if len(union) > 0:
                return len(intersection)/len(union)

        if isinstance(email2, list) and not isinstance(email1, list):
            if email1 in set(email2):
                return 1
        if isinstance(email1, list) and not isinstance(email2, list):
            if email2 in set(email1):
                return 1
        if email1 == email2:
            return 1
        return 0

    @staticmethod
    def get_score_for_name(row):
        return 1 \
            if row['FIRST_NAME1'] == row['FIRST_NAME2'] and \
            row['LAST_NAME1'] == row['LAST_NAME2'] \
            else 0

    @staticmethod
    def get_institute_name(institutions, author):
        """
        This method finds the institute name to which the author belongs.

        If we look at a training record and specifically the 'INSTITUTE' field, We have 3 different cases here
        CASE 1. Institute name is a String, For example :
                        "Univ Kansas, Med Ctr, Kansas City, KS 66160 USA."

        CASE 2. Institute name is a List, For example:
                        [u'MARINE BIOL LAB,WOODS HOLE,MA.', u'UNIV MASSACHUSETTS,AMHERST,MA.',
                        u'REED COLL,PORTLAND,OR.', u'UNIV CONNECTICUT,BIOL SCI GRP,STORRS,CT 06268.']

        CASE 3. Institutions name is a map, where each author is mapped to his/her institute. For example:
                        [u'[Telfer, Evelyn E.] Univ Edinburgh, Inst Cell Biol, Edinburgh, Midlothian, Scotland.',
                        u'[Telfer, Evelyn E.] Univ Edinburgh, Ctr Integrat Physiol, Edinburgh, Midlothian, Scotland.',
                        u'[Albertini, David F.] Univ Kansas, Med Ctr, Inst Reprod Hlth & Regenerat Med, Ctr Reprod Sci,
                                                                                            Kansas City, KS 66103 USA.']

        We deal with the above scenarios in the following way.

        For CASE 1 : Return the institute name as-is.

        For CASE 2 : No way to link author to its institute. So return ``None``

        For CASE 3 : We try to find the correct mapping. If there is match, we return the found institute name

        :param institutions:
        :param author:
        :return:
        """
        WORD = re.compile(r'\w+')
        if type(institutions) is str:
            if institutions.startswith('[') and institutions.endswith(']'):
                    a = literal_eval(institutions)
                    for entry in a:
                        m = re.search(r"\[(.*?)\]", entry)
                        if m is not None:
                            author_name = m.group(1)
                            tokens = set(x.lower() for x in WORD.findall(author_name))
                            intersection = set([x.lower() for x in author]) & tokens
                            n = re.search(r"(?<=\]).*", entry)
                            if n is not None:
                                institute_name = n.group(0)
                            if len(intersection) > 0:
                                return institute_name
            elif not institutions.startswith('[') and not institutions.endswith(']'):
                    return institutions

    @staticmethod
    def get_score_for_institute_names(row):
        '''
        For the training record, passed in input.
        1. GET INSTITUTE 1 using the method get_institute_name()
        2. GET INSTITUTE 2
        3. Return a cosine similarity score in between the 2 institute names.

        Please read https://diging.atlassian.net/wiki/pages/viewpage.action?pageId=46432257 for more details.

        :param training_record:
        :return: A score between 0 and 1.
        '''
        institute1 = Compare.get_institute_name(row['INSTITUTE1'], row['LAST_NAME1'])
        institute2 = Compare.get_institute_name(row['INSTITUTE2'], row['LAST_NAME2'])
        if institute1 is not None and institute2 is not None:
            institute1 = join_institute_names(split_institute(institute1)[0:3])
            institute2 = join_institute_names(split_institute(institute2)[0:3])
            return cosine_similarity(sentence_to_vector(institute1), sentence_to_vector(institute2))
        return 0

    def calculate_scores(self):
        """
        Calculate the scores for each feature defined below.

        >>> features = ['INSTIT_SCORE','BOTH_NAME_SCORE','FNAME_SCORE','FNAME_PARTIAL_SCORE','LNAME_SCORE',
        >>> 'LNAME_PARTIAL_SCORE','EMAIL_ADDR_SCORE','AUTH_KW_SCORE','COAUTHOR_SCORE','MATCH']

        :return:
        """
        self.record_df['INSTIT_SCORE'] = self.record_df.apply(lambda row: Compare.get_score_for_institute_names(row), axis=1)
        self.record_df['BOTH_NAME_SCORE'] = self.record_df.apply(lambda row: Compare.get_score_for_name(row), axis=1)
        self.record_df['FNAME_SCORE'] = self.record_df.apply(lambda row: max(fuzz.ratio(row['FIRST_NAME1'], row['FIRST_NAME2'])/100.0, fuzz.ratio(row['FIRST_NAME2'], row['FIRST_NAME1'])/100.0), axis=1)
        self.record_df['LNAME_SCORE'] = self.record_df.apply(lambda row: max(fuzz.ratio(row['LAST_NAME1'], row['LAST_NAME2'])/100.0, fuzz.ratio(row['LAST_NAME2'], row['LAST_NAME1'])/100.0), axis=1)
        self.record_df['LNAME_PARTIAL_SCORE'] = self.record_df.apply(lambda row: max(fuzz.partial_ratio(row['LAST_NAME1'], row['LAST_NAME2'])/100.0, fuzz.partial_ratio(row['LAST_NAME2'], row['LAST_NAME1'])/100.0), axis=1)
        self.record_df['FNAME_PARTIAL_SCORE'] = self.record_df.apply(lambda row: max(fuzz.partial_ratio(row['FIRST_NAME1'], row['FIRST_NAME2'])/100.0, fuzz.partial_ratio(row['FIRST_NAME2'], row['FIRST_NAME2'])/100.0), axis=1)
        self.record_df['EMAIL_ADDR_SCORE'] = self.record_df.apply(lambda row: Compare.get_score_for_email_address(row), axis=1)
        self.record_df['AUTH_KW_SCORE'] = self.record_df.apply(lambda row: Compare.get_score_for_author_keywords(row), axis=1)
        self.record_df['COAUTHOR_SCORE'] = self.record_df.apply(lambda row: Compare.get_score_for_coauthors(row), axis=1)
        self.scores_df = self.record_df[features]


def classify(paper_sample1, paper_sample2):
    compare_instance = Compare(paper_sample1, paper_sample2)
    compare_instance.create_single_record()
    compare_instance.calculate_scores()
    return clf.predict(compare_instance.scores_df[features])









