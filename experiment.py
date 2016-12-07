import csv
import datetime
import json
import random
from collections import defaultdict

import numpy as np

import evaluation
import recommendation
import util
from network import Network
from util import data_path, out_path, print_error, with_prob
from article import Article


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


class Experiment(object):

    SOURCES = PLike.TRUST.keys()
    WEIGHTS_SOURCES = [1.0 / len(SOURCES)] * len(SOURCES)  # uniform weights

    def __init__(self,
                 num_iterations=100,
                 all_analyses=False,
                 recommender='Random',
                 networkInitType='1',
                 pLikeMethod='empirical',
                 friendGraphFile='zacharys.csv',
                 numRecsPerIteration=1,
                 ):
        """
        Constructor for Experiment.

        :param num_iterations: int number of iterations to run in this experiment.
        :param all_analyses: True to run all the analyses, False if not
        :param recommender: string name of the recommender class to use as defined in recommender.py
        :param pLikeMethod: method name of the PLike version to use: 'uniform'|'empirical'
        :param friendGraphFile: filename of the friend graph CSV file to use
        """
        self.start_time = datetime.datetime.now()
        self.num_iterations = num_iterations
        self.all_analyses = all_analyses
        self.numRecsPerIteration = numRecsPerIteration
        self.network = Network(data_path(friendGraphFile), networkInitType)
        self.pLike = getattr(PLike, pLikeMethod)
        self.recommender = getattr(recommendation, recommender)()
        self.metrics = []
        if self.all_analyses:
            self.metrics = [
                evaluation.ReadingDistribution(),
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
                evaluation.EigenVectors(),
                evaluation.MoreEigenVectors(),
                evaluation.CommonArticles(-2, 2),
                evaluation.CommonArticles(-1, 2),
                evaluation.CommonArticles(-2, 1),
                evaluation.CommonArticles(1,2),
                evaluation.CommonArticles(2,2),
                evaluation.CommonArticles(-2, -2),
                evaluation.Betweenness(),
                evaluation.ModularityWRTFriends(),
                evaluation.BetweennessWRTFriends(),
                evaluation.OverallClusteringWRTFriends(),
                evaluation.ClusterPoliticalnessWRTFriends("all"),
                evaluation.EigenVectorsWRTFriends(),
                evaluation.MoreEigenVectorsWRTFriends(),
            ]
        else:
            self.metrics = [evaluation.GraphViz()]  # evaluation.Statistics()]
        self.histories = defaultdict(list)

    def introduceArticle(self, iterations):
        # Create an article from a random source
        article = Article(np.random.choice(self.SOURCES, p=self.WEIGHTS_SOURCES))
        article.incrementTimeToLive(iterations)
        self.network.addArticle(article)
        return article

    def runRecommendation(self, readers):
        # Compute recommendations and "show" them to users
        allRecs = self.recommender.makeRecommendations(self.network, readers, N=self.numRecsPerIteration)
        for readerId, recs in allRecs.iteritems():
            reader = self.network.getUser(readerId)
            for recommendedArticle in recs:
                if random.random() < self.pLike(reader, recommendedArticle):
                    self.network.addEdge(reader, recommendedArticle)

    def runAnalysis(self, iterations):
        for metric in self.metrics:
            self.histories[metric].append(metric.measure(self.network, iterations))

    def killArticles(self, iterations):
        for article in self.network.articles.itervalues():
            if not article.getIsDead() and article.getTimeToLive() < iterations:
                article.setIsDead(True)
    
    def run(self):
        # Create an initial set of articles
        # (At least as many as we recommend per iteration,
        # if not several times more.)
        for _ in xrange(self.numRecsPerIteration * 4):
            self.introduceArticle(0)

        for i in util.visual_xrange(self.num_iterations, use_newlines=False):
            self.step(i)

    def step(self, i):
        """Perform one step of the simulation."""
        readers = self.network.getNextReaders()  # get readers that can read at this time point

        # Introduce a new article
        self.introduceArticle(i)

        # Compute recommendations and "show" them to users
        self.runRecommendation(readers)

        # Analyze resulting graph at this iteration
        self.runAnalysis(i)

        # Kill articles that have reached their lifetime
        self.killArticles(i)

    def saveResults(self):
        # Save results
        for metric in self.metrics:
            metric.save(self.histories[metric])

        # Try to plot
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print_error("matplotlib not available, skipping plots")

        for metric in self.metrics:
            metric.plot(self.network, self.histories[metric])


def runExperiment(*args, **kwargs):
    """
    Create a new Experiment and run the full simulation and save the results.

    All arguments are passed as-is to the Experiment constructor.

    Example usage in the Python console:
        >>> import experiment
        >>> experiment.runExperiment(num_iterations=10, recommender='Random')

    To save you the time of opening a Python console, you can do this in one line from the shell:
        # python -c "import experiment; experiment.runExperiment(num_iterations=10, recommender='Random')"
    """
    exp = Experiment(*args, **kwargs)
    exp.run()
    exp.saveResults()

    # Save parameters
    with open(out_path('parameters.json'), 'wb') as fp:
        json.dump(kwargs, fp)


if __name__ == "__main__":
    runExperiment()
    # http://www.slideshare.net/koeverstrep/tutorial-bpocf
