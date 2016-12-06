import util
import random
from network import Network
from articleGenerator import ArticleGenerator
import evaluation
import recommendation
from util import print_error, data_path, out_path
from collections import defaultdict
from itertools import izip
import pdb

class Experiment(object):

    SOURCES = ["NYTimes", "WSJ", "Fox"]
    WEIGHTS_SOURCES = [1.0/3, 1.0/3, 1.0/3]
    NUM_SIMULATIONS = 500

    def __init__(self):
        self.articleGenerators = []
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[0], [.1, .3, 0, .3, .1]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[1], [0, .2, .5, .3, 0]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[2], [.7, .2, .1, 0, 0]))
        self.network = Network()
        self.recommender = recommendation.RandomRecommender()
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
            evaluation.ClusterPolticalness("-2"),
            evaluation.ClusterPolticalness("-1"),
            evaluation.ClusterPolticalness("0"),
            evaluation.ClusterPolticalness("1"),
            evaluation.ClusterPolticalness("2"),
            evaluation.ClusterPolticalness("all"),
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
            evaluation.ModularityWRTFriends(),
            evaluation.BetweennessWRTFriends(),
            evaluation.OverallClusteringWRTFriends(),
            evaluation.ClusterPolticalnessWRTFriends("all"),
            #evaluation.EigenVectorsWRTFriends(),
            #evaluation.MoreEigenVectorsWRTFriends(),
        ]
        self.histories = defaultdict(list)

    def createArticle(self):
        idx = util.generatePoliticalness(self.WEIGHTS_SOURCES)
        articleGen = self.articleGenerators[idx]
        return articleGen.createArticle()

    def PLikeBySource(self, reader, article):
        if reader.getPoliticalness() < 0 and article.getSource() == self.SOURCES[2]:
            return .9
        if reader.getPoliticalness() == 0 and article.getSource() == self.SOURCES[1]:
            return .8
        return self.PLike(reader, article)

    def PLike(self, reader, article):
        diff = abs(reader.getPoliticalness() - article.getPoliticalness())
        diffToProb = {
            0: .6,
            1: .4,
            2: .2,
            3: .1,
            4: .1,
        }
        return diffToProb[diff]

    def triadicClosureBasedOnFriends(self, iterations, force = True):
        article = self.createArticle()
        article.incrementTimeToLive(iterations)
        self.network.addArticle(article)
        randReaders = random.sample(self.network.users.keys(), 1)
        for reader in randReaders:
            probLike = self.PLikeBySource(self.network.users[reader], article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(self.network.users[reader], article)
                neighbors = self.network.friendGraph.GetNI(reader).GetOutEdges()
                neighs = []
                for n in neighbors:
                    neighs.append(n)
                randNeighbor = random.sample(neighs, 1)
                #can either force neighrbot to read it ord test with pLike
                if force:
                    self.network.addEdge(self.network.getUser(randNeighbor[0]), article)
                else:
                    if self.PLikeBySource(self.network.getUser(randNeighbor[0]), article) < random.random():
                        self.network.addEdge(self.network.getUser(randNeighbor[0]), article)
        readers = self.network.users.values()
        allRecs = self.recommender.makeRecommendations(self.network, readers, N=1)
        for readerId, recs in allRecs.iteritems():
            reader = self.network.getUser(readerId)
            for recommendedArticle in recs:
                if random.random() < self.PLikeBySource(reader, recommendedArticle):
                    self.network.addEdge(reader, recommendedArticle)
        #self.help0DegreeUsers(iterations, article)
        #self.help0DegreeArticles(iterations, self.network.users.values())
        self.runAnalysis(iterations)


    def randomRandomCompleteTriangles(self, iterations):
        article = self.createArticle()
        article.incrementTimeToLive(iterations)
        self.network.addArticle(article)
        randReaders = random.sample(self.network.users.keys(), 1)
        for reader in randReaders:
            probLike = self.PLike(self.network.users[reader], article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(self.network.users[reader], article)
                neighbors = self.network.userArticleGraph.GetNI(reader).GetOutEdges()
                neighs = []
                for n in neighbors:
                    neighs.append(n)
                rand = random.sample(neighs, 1)
                print rand
                #rand is an article
                users = []
                for r in rand:
                    neighbors = self.network.userArticleGraph.GetNI(r).GetOutEdges()
                    for n in neighbors:
                        users.append(n)
                rand = random.sample(users, 1)
                for r in rand:
                    self.network.addEdge(self.network.users[r], article)
        
        self.runAnalysis(iterations)

    def help0DegreeUsers(self, iterations, article, N=5):
        if iterations % N == 0:
            users = self.network.getUsersWithDegree0()
            for u in users:
                probLike = self.PLike(u, article)
                if random.random() < probLike:
                    self.network.addEdge(u, article)

    def help0DegreeArticles(self, iterations, users, N=4):
        if iterations % N == 0:
            articles = self.network.getArticlesWithDegree0()
            for a in articles:
                for u in users:
                    probLike = self.PLike(u, a)
                    if random.random() < probLike:
                        self.network.addEdge(u, a)


    def forceConnectedGraph(self, iterations, article):
        if iterations == 0:
            readers = self.network.users.values()
            for reader in readers:
                self.network.addEdge(reader, article)

    def simulate(self, iterations, all_analyses=True):
        readers = self.network.getNextReaders() # get readers that can read at this time point

        # Introduce a new article
        article = self.createArticle()
        article.incrementTimeToLive(iterations)
        self.network.addArticle(article)
        #self.forceConnectedGraph(iterations, article)
        for reader in readers: # ask each reader if like it or not
            probLike = self.PLike(reader, article)
            if random.random() < probLike:
                self.network.addEdge(reader, article)

        # Compute recommendations and "show" them to users
        allRecs = self.recommender.makeRecommendations(self.network, readers, N=1)
        for readerId, recs in allRecs.iteritems():
            reader = self.network.getUser(readerId)
            for recommendedArticle in recs:
                if random.random() < self.PLike(reader, recommendedArticle):
                    self.network.addEdge(reader, recommendedArticle)

        # On every third iteration, "show" the readers the top 5 most popular articles
        if iterations % 3 == 0:
            articleDeg = evaluation.getArticleDegreeDistribution(self.network, 'alive')
            sortedDeg = sorted(articleDeg, key=lambda x: x[1], reverse=True)
            topFive = sortedDeg[0:5]
            for (aId, _) in topFive:
                article = self.network.getArticle(aId)
                for reader in readers:
                    probLike = self.PLike(reader, article)
                    if random.random() < probLike:
                        self.network.addEdge(reader, article)

        # if iterations % 5 == 0:
        #     users = self.network.getUsersWithDegree0()
        #     for u in users:
        #         probLike = self.PLike(u, article)
        #         if random.random() < probLike:
        #             self.network.addEdge(u, article)
        self.runAnalysis(iterations, all_analyses)

    def runAnalysis(self, iterations, all=True):
        if all:
            for metric in self.metrics:
                self.histories[metric].append(metric.measure(self.network, iterations))
        else:
            metricsToRun = [evaluation.ReadingDistribution()] #evaluation.Statistics()]
            for metric in metricsToRun:
                self.histories[metric].append(metric.measure(self.network, iterations))


    def killArticles(self, iterations):
        for article in self.network.articles.itervalues():
            #print article
            if not article.getIsDead() and article.getTimeToLive() < iterations:
                article.setIsDead(True)
    
    def runAllSimulation(self, all_analyses=True):
        for i in util.visual_xrange(self.NUM_SIMULATIONS, use_newlines=False):
            self.simulate(100, all_analyses)
            # self.triadicClosureBasedOnFriends(i)
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
    exp.runAllSimulation(all_analyses=False)
    exp.saveResults()
