import datetime
import json
import os
import random
from collections import defaultdict

import numpy as np

import evaluation
import recommendation
import util
from article import Article
from network import Network
from util import data_path, out_path, print_error, with_prob


class PLike(object):
    UNIFORM_LIKE_PROB = 0.2

    TRUST = util.load_trust_data()

    @staticmethod
    def uniform(reader, article):
        return PLike.UNIFORM_LIKE_PROB

    @staticmethod
    def empirical(reader, article):
        # Like probability is equal to the trust percentage from the data.
        # This doesn't always make sense though: very few people "trust"
        # BuzzFeed, but a lot of people still like and share their listicles.

        return PLike.TRUST[article.source][reader.politicalness]

    @staticmethod
    def individual(reader, article):
        # Like probability is equal to a combination of the trust percentages from the data for that source
        # a*plikeofreader with diff politcalness
        PLike_for_source = PLike.TRUST[article.source]
        # print PLike_for_source
        weighting = np.array([0]*5)
        for i in PLike_for_source.keys():
            weighting[i] = np.random.normal((10/(np.absolute(i-reader.politicalness)+1)),
                                            (1/(np.absolute(reader.politicalness)+0.5)))

        '''
        print weighting
        print 'actual'
        print PLike_for_source[reader.politicalness]
        print 'indiv'
        print (np.array(PLike_for_source.values()).dot(weighting))/np.sum(weighting)
        '''

        return (np.array(PLike_for_source.values()).dot(weighting))/np.sum(weighting)#PLike.TRUST[article.source][reader.politicalness]



class Experiment(object):

    SOURCES = PLike.TRUST.keys()
    WEIGHTS_SOURCES = [1.0 / len(SOURCES)] * len(SOURCES)  # uniform weights

    def __init__(self,
                 numIterations=100,
                 allAnalyses=False,
                 recommender='CollaborativeFiltering',
                 nullRecommender='Random',
                 networkInitType='random',
                 pLikeMethod='empirical',
                 friendGraphFile='CA-GrQc.txt',
                 numOnlinePerIteration=100,
                 numRecsPerIteration=5,
                 ):
        """
        Constructor for Experiment.

        :param numIterations: int number of iterations to run in this experiment.
        :param allAnalyses: True to run all the analyses, False if not
        :param recommender: string name of the recommender class to use as defined in recommender.py
        :param pLikeMethod: method name of the PLike version to use: 'uniform'|'empirical'
        :param friendGraphFile: filename of the friend graph CSV file to use
        """
        self.start_time = datetime.datetime.now()
        self.numIterations = numIterations
        self.allAnalyses = allAnalyses
        self.numRecsPerIteration = numRecsPerIteration
        self.numOnlinePerIteration = numOnlinePerIteration
        self.networkInitType = networkInitType
        self.network = Network(data_path(friendGraphFile), networkInitType)
        self.pLikeMethod = pLikeMethod
        self.pLike = getattr(PLike, pLikeMethod)
        self.recommender = getattr(recommendation, recommender)()
        self.nullRecommender = getattr(recommendation, nullRecommender)()
        if self.allAnalyses:
            self.metrics = [
                #evaluation.ReadingDistribution(),
                evaluation.PathsBetweenPoliticalnesses(),
                evaluation.PathsBetweenPoliticalnesses(-1, 1),
                evaluation.PathsBetweenPoliticalnesses(-2, -1),
                evaluation.PathsBetweenPoliticalnesses(1, 2),
                evaluation.PathsBetweenPoliticalnesses(2, 2),
                evaluation.PathsBetweenPoliticalnesses(-2, -2),
                evaluation.UserDegreeDistribution("all"),
                evaluation.Modularity(),
                evaluation.ArticleDegreeDistribution("all"),
                evaluation.ArticleDegreeDistribution("alive"),
                evaluation.ArticleDegreeDistribution("dead"),
                evaluation.DistributionOfLifeTime("alive"),
                evaluation.AliveArticles(),
                evaluation.DeadArticles(),
                evaluation.OverallClustering(),
                evaluation.ClusterPoliticalness("-2"),
                evaluation.ClusterPoliticalness("-1"),
                evaluation.ClusterPoliticalness("0"),
                evaluation.ClusterPoliticalness("1"),
                evaluation.ClusterPoliticalness("2"),
                evaluation.ClusterPoliticalness("all"),
                evaluation.LargestConnectedComponent(),
                #evaluation.EigenVectors(),
                #evaluation.MoreEigenVectors(),
                evaluation.CommonArticles(-2, 2),
                evaluation.CommonArticles(-1, 2),
                evaluation.CommonArticles(-2, 1),
                evaluation.CommonArticles(1,2),
                evaluation.CommonArticles(2,2),
                evaluation.CommonArticles(-2, -2),
                evaluation.Betweenness(),
            ]
        else:
            self.metrics = [
                evaluation.Statistics() #,
                #evaluation.UserUserGraphCutMinimization(),
            ]

        self.histories = defaultdict(list)

    def _parameters(self, delimiter):
        return delimiter.join([
            "%s Recommender" % self.recommender.__class__.__name__,
            "%s Null Recommender" % self.nullRecommender.__class__.__name__,
            "%s Political Preference" % self.networkInitType.title(),
            "%s PLike" % self.pLikeMethod.title(),
            "%d NumOnline" % self.numOnlinePerIteration,
            "%d NumRecs" % self.numRecsPerIteration,
            "%d Iterations" % self.numIterations,
        ])

    @property
    def parameters(self):
        return self._parameters(delimiter=', ')

    def out_path(self, filename):
        subdir = self._parameters(delimiter='.').replace(' ', '')
        return out_path(filename, subdir=subdir)

    def introduceArticle(self, iterations):
        # Create an article from a random source
        article = Article(np.random.choice(self.SOURCES, p=self.WEIGHTS_SOURCES), self.numIterations, iterations)
        self.network.addArticle(article)
        return article

    def showRecommendations(self, recommender, readers, N):
        allRecs = recommender.makeRecommendations(self.network, readers, N=N)
        for readerId, recs in allRecs.iteritems():
            reader = self.network.getUser(readerId)
            for recommendedArticle in recs:
                if with_prob(self.pLike(reader, recommendedArticle)):
                    self.network.addEdge(reader, recommendedArticle)

    def runAnalysis(self, iterations):
        for metric in self.metrics:
            self.histories[metric].append(metric.measure(self, self.network, iterations))

    def run(self):
        # Create an initial set of articles
        # (At least as many as we recommend per iteration,
        # if not several times more.)
        for _ in xrange(self.numRecsPerIteration * 4):
            self.introduceArticle(0)

        for i in util.visual_xrange(self.numIterations):
            self.step(i)

        self.network.removeUnlikedArticles()

    def step(self, i):
        """Perform one step of the simulation."""
        # get readers that can read at this time point
        readers = self.network.getNextReaders(self.numOnlinePerIteration)

        # Introduce new articles
        new_articles = [self.introduceArticle(i) for _ in xrange(self.numRecsPerIteration)]

        # print "%f%% ARTICLES LIKED, NUM READERS: %d" % (
        #     sum(self.network.userArticleGraph.GetNI(article.articleId).GetDeg() > 0 for article in self.network.getArticles()) /
        #     sum(1. for _ in self.network.getArticles()),
        #     len(readers)
        # )

        # Compute recommendations and "show" them to users
        self.showRecommendations(self.nullRecommender, readers, self.numRecsPerIteration / 2)
        self.showRecommendations(self.recommender, readers, self.numRecsPerIteration / 2)

        # Analyze resulting graph at this iteration
        self.runAnalysis(i)

        # Remove justAdded flag from new articles
        for article in new_articles:
            article.justAdded = False

        # Kill articles that have reached their lifetime
        self.network.reapArticles(i)

    def saveResults(self):
        # Save results
        for metric in self.metrics:
            metric.save(self, self.histories[metric])

        # Try to plot
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print_error("matplotlib not available, skipping plots")

        for metric in self.metrics:
            metric.plot(self, self.network, self.histories[metric])


def runExperiment(*args, **kwargs):
    """
    Create a new Experiment and run the full simulation and save the results.

    All arguments are passed as-is to the Experiment constructor.

    Example usage in the Python console:
        >>> import experiment
        >>> experiment.runExperiment(numIterations=10, recommender='Random')

    To save you the time of opening a Python console, you can do this in one line from the shell:
        python -c "import experiment; experiment.runExperiment(numIterations=10, recommender='Random')"
    """
    exp = Experiment(*args, **kwargs)
    exp.run()
    exp.saveResults()

    # Save parameters
    with open(exp.out_path('parameters.json'), 'wb') as fp:
        json.dump(kwargs, fp)


if __name__ == "__main__":
    runExperiment()
    # http://www.slideshare.net/koeverstrep/tutorial-bpocf
