import csv
import datetime
import json
import random
from collections import defaultdict

import numpy as np

import evaluation
import recommendation
import util
from articleGenerator import ArticleGenerator
from network import Network
from util import data_path, out_path, print_error


class PLike(object):
    @staticmethod
    def basedOnData(reader, article):
        data = []
        with open(data_path("percof-readers-trust.csv")) as f:
            csvreader = csv.reader(f, delimiter=",")
            for row in csvreader:
                oneRow = []
                for col in row:
                    oneRow.append(col)
                data.append(oneRow)

        userPoliticalness = reader.getPoliticalness()
        userPoliticalnessToDataIndex = {2 : 2, 1 : 3, 0 : 4, -1 : 5, -2 : 6}
        source = article.getSource()
        for row in data:
            if row[0] == source:
                colIndex = userPoliticalnessToDataIndex[userPoliticalness]
                return row[0][colIndex]
        raise Exception("Invalid Article Source")

    @staticmethod
    def bySource(reader, article):
        if reader.getPoliticalness() < 0 and article.getSource() == Experiment.SOURCES[2]:
            return .9
        if reader.getPoliticalness() == 0 and article.getSource() == Experiment.SOURCES[1]:
            return .8
        return PLike(reader, article)

    @staticmethod
    def static(reader, article):
        diff = abs(reader.getPoliticalness() - article.getPoliticalness())
        diffToProb = {
            0: .6,
            1: .4,
            2: .2,
            3: .1,
            4: .1,
        }
        return diffToProb[diff]


class Experiment(object):

    SOURCES = ["New York Times", "Wall Street Journal", "Fox News"]
    WEIGHTS_SOURCES = [1.0/3, 1.0/3, 1.0/3]

    def __init__(self,
                 num_iterations=500,
                 all_analyses=False,
                 recommender='Random',
                 networkInitType='1',
                 pLikeMethod='basedOnData',
                 numRecsPerIteration=1,
                 force=True,  # old
                 simulation="simulate",  # old
                 help0DegreeUsers=False,  # old
                 help0DegreeArticles=False,  # old
                 popular=True,  #old
                 ) :
        """
        Constructor for Experiment.

        :param num_iterations: int number of iterations to run in this experiment.
        :param all_analyses: True to run all the analyses, False if not
        :param simulation: IGNORED
        :param recommender: string name of the recommender class to use as defined in recommender.py
        :param pLikeMethod: method name of the PLike version to use
        """
        self.start_time = datetime.datetime.now()
        self.num_iterations = num_iterations
        self.all_analyses = all_analyses
        self.numRecsPerIteration = numRecsPerIteration
        self.articleGenerators = []
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[0], [.15, .35, 0, .35, .15]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[1], [0, .2, .5, .3, 0]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[2], [.7, .2, .1, 0, 0]))
        self.network = Network(networkInitType)
        self.pLike = getattr(PLike, pLikeMethod)
        self.recommender = vars(recommendation)[recommender]()
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
            self.metrics = [evaluation.ReadingDistribution()]
                            #evaluation.Statistics()]
        self.histories = defaultdict(list)

    def introduceArticle(self, iterations):
        articleGen = np.random.choice(self.articleGenerators, p=self.WEIGHTS_SOURCES)
        article = articleGen.createArticle()
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
        for i in util.visual_xrange(self.num_iterations, use_newlines=False):
            self.step(i)

    def step(self, i):
        """Perform one step of the simulation."""
        readers = self.network.getNextReaders()  # get readers that can read at this time point

        # Introduce a new article
        article = self.introduceArticle(i)

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
