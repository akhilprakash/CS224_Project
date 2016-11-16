import util
import random
from network import Network
from articleGenerator import ArticleGenerator
from recommendation import RandomRecommender
import evaluation
from util import print_error, data_path, out_path
from collections import defaultdict
import pdb

class Experiment(object):

    SOURCES = ["NYTimes", "WSJ", "Fox"]
    WEIGHTS_SOURCES = [1.0/3, 1.0/3, 1.0/3]
    NUM_SIMULATIONS = 100

    def __init__(self):
        self.articleGenerators = []
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[0], [.1, .3, 0, .3, .1]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[1], [0, .2, .5, .3, 0]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[2], [.7, .2, .1, 0, 0]))
        self.network = Network()
        self.recommender = RandomRecommender()
        self.distributionResults = []
        self.pathResults = []
        self.userDegreeDistribution = []
        self.articleDegreeDistribution = []
        self.aliveArticleDegreeDistribution = []
        self.deadArticleDegreeDistribution = []
        self.lifeTimeDistribution = []
        self.metrics = [
            evaluation.ReadingDistribution(),
            evaluation.PathsBetweenPoliticalnesses(),
            evaluation.UserDegreeDistribution("all"),
            evaluation.Modularity(),
            evaluation.ArticleDegreeDistribution("all"),
            evaluation.ArticleDegreeDistribution("alive"),
            evaluation.ArticleDegreeDistribution("dead"),
            evaluation.DistributionOfLifeTime(),
        ]
        self.histories = defaultdict(list)

    def createArticle(self):
        idx = util.generatePoliticalness(self.WEIGHTS_SOURCES)
        articleGen = self.articleGenerators[idx]
        return articleGen.createArticle()

    def PLike(self, reader, article):
        diff = abs(reader.getPoliticalness() - article.getPoliticalness())
        diffToProb = {0:.6, 1:.4, 2:.2, 3:.1, 4:.1}
        return diffToProb[diff]

    def randomRandomCompleteTriangles(self, iterations):
        article = self.createArticle()
        article.incrementTimeToLive(iterations)
        self.network.addArticle(article)
        randReaders = random.sample(self.network.users, 1)
        for reader in randReaders:
            probLike = self.PLike(reader, article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(reader, article)
                neighbors = self.network.getOutEdges(reader.getUserId())
                rand = random.sample(neighbors, 1)
                for r in rand:
                    self.network.addEdge(r, article)
        self.runAnalysis()

    def simulate(self, iterations):
        article = self.createArticle()
        #pdb.set_trace()
        article.incrementTimeToLive(iterations)
        readers = self.network.getNextReaders()
        self.network.addArticle(article)

        for reader in readers:
            rec = self.recommender.makeRecommendations(self.network, reader)
            # TODO: do something
            #recommend to readers
            #see if readers like
            #if it does add edge

        for reader in readers:
            probLike = self.PLike(reader, article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(reader, article)

        if iterations % 3 == 0:
            articleDeg = evaluation.getArticleDegreeDistribution(self.network, 'alive')
            sortedDeg = sorted(articleDeg, key=lambda x: x[1], reverse=True)
            topFive = sortedDeg[0:5]
            for (aId, _) in topFive:
                article = self.network.getArticle(aId)
                for reader in readers:
                    probLike = self.PLike(reader, article)
                    rand = random.random()
                    if rand < probLike:
                        self.network.addEdge(reader, article)

        self.runAnalysis(iterations)

    def runAnalysis(self, iterations):
        for metric in self.metrics:
            self.histories[metric].append(metric.measure(self.network, iterations))

    def killArticles(self, iterations):
        for article in self.network.articles.itervalues():
            #print article
            if not article.getIsDead() and article.getTimeToLive() < iterations:
                article.setIsDead(True)
                print "killed article Id = " + str(article.getArticleId())
    
    def runAllSimulation(self):
        for i in util.visual_xrange(self.NUM_SIMULATIONS, use_newlines=True):
            self.simulate(i)
            self.killArticles(i)
        #print self.distributionResults

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
            metric.plot(self.histories[metric])

if __name__ == "__main__":
    exp = Experiment()
    exp.runAllSimulation()
    exp.saveResults()
