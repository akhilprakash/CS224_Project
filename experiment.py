import util
import random
from network import Network
from articleGenerator import ArticleGenerator
import evaluation
import recommendation
from util import print_error, data_path, out_path
from collections import defaultdict
import csv
import datetime
import json


def PLikeBaseOnData(reader, article):
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

def PLikeBySource(reader, article):
    if reader.getPoliticalness() < 0 and article.getSource() == self.SOURCES[2]:
        return .9
    if reader.getPoliticalness() == 0 and article.getSource() == self.SOURCES[1]:
        return .8
    return self.PLike(reader, article)

def PLike(reader, article):
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
                 simulation="simulate",
                 recommender='RandomRecommender',
                 initialize="1",
                 force=True,
                 plike="data",
                 help0DegreeUsers=False,
                 help0DegreeArticles=False,
                 popular=True):
        """
        Constructor for Experiment.

        :param num_iterations: int number of iterations to run in this experiment.
        :param all_analyses: True to run all the analyses, False if not
        :param recommender: string name of the recommender class to use as defined in recommender.py
        """
        self.start_time = datetime.datetime.now()
        self.num_iterations = num_iterations
        self.all_analyses = all_analyses
        self.simulation = simulation
        self.force = force
        self.shouldHelp0DegreeUsers = help0DegreeUsers
        self.shouldHelp0DegreeArticles = help0DegreeArticles
        self.plike = PLikeBaseOnData if plike == "data" else (PLikeBySource if "source" else plike)
        self.popular = popular
        self.initialize = initialize
        self.articleGenerators = []
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[0], [.1, .3, 0, .3, .1]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[1], [0, .2, .5, .3, 0]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[2], [.7, .2, .1, 0, 0]))
        self.network = Network(initialize)
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

    def createArticle(self):
        idx = util.generatePoliticalness(self.WEIGHTS_SOURCES)
        articleGen = self.articleGenerators[idx]
        return articleGen.createArticle()

    def introduceArticle(self, iterations):
        article = self.createArticle()
        article.incrementTimeToLive(iterations)
        self.network.addArticle(article)
        return article

    def triadicClosureBasedOnFriends(self, iterations, force = True, plike = PLikeBaseOnData):
        article = self.introduceArticle(iterations)
        randReaders = random.sample(self.network.users.keys(), 1)
        for reader in randReaders:
            probLike = plike(self.network.users[reader], article)
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
                    if plike(self.network.getUser(randNeighbor[0]), article) < random.random():
                        self.network.addEdge(self.network.getUser(randNeighbor[0]), article)
        readers = self.network.users.values()
        self.runRecommendation(readers, plike)
        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())
        self.runAnalysis(iterations)

    def randomRandomCompleteTriangles(self, iterations, plike=PLike):
        article = self.introduceArticle(iterations)
        randReaders = random.sample(self.network.users.keys(), 1)
        for reader in randReaders:
            probLike = plike(self.network.users[reader], article)
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
        self.runRecommendation(randReaders, plike)
        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())
        self.runAnalysis(iterations)

    def help0DegreeUsers(self, iterations, article, plike=PLikeBaseOnData, N=5):
        if iterations % N == 0:
            users = self.network.getUsersWithDegree0()
            for u in users:
                probLike = plike(u, article)
                if random.random() < probLike:
                    self.network.addEdge(u, article)

    def help0DegreeArticles(self, iterations, users, plike=PLikeBaseOnData, N=4):
        if iterations % N == 0:
            articles = self.network.getArticlesWithDegree0()
            for a in articles:
                for u in users:
                    probLike = plike(u, a)
                    if random.random() < probLike:
                        self.network.addEdge(u, a)

    def forceConnectedGraph(self, iterations, article):
        if iterations == 0:
            readers = self.network.users.values()
            for reader in readers:
                self.network.addEdge(reader, article)

    def recc_system_simulate(self, iterations, plike = PLikeBaseOnData):
        readers = self.network.getNextReaders()  # get readers that can read at this time point

        # Introduce a new article
        article = self.introduceArticle(iterations)
        # self.forceConnectedGraph(iterations, article)

        # Compute recommendations and "show" them to users
        self.runRecommendation(readers, plike)

        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())

        # Analyze resulting graph at this iteration
        self.runAnalysis(iterations)


    def runRecommendation(self, readers, plike=PLikeBaseOnData):
        # Compute recommendations and "show" them to users
        allRecs = self.recommender.makeRecommendations(self.network, readers, N=1)
        for readerId, recs in allRecs.iteritems():
            reader = self.network.getUser(readerId)
            for recommendedArticle in recs:
                if random.random() < plike(reader, recommendedArticle):
                    self.network.addEdge(reader, recommendedArticle)


    def simulate(self, iterations, plike=PLikeBaseOnData):
        readers = self.network.getNextReaders() # get readers that can read at this time point

        # Introduce a new article
        article = self.introduceArticle(iterations)

        for reader in readers: # ask each reader if like it or not
            probLike = plike(reader, article)
            if random.random() < probLike:
                self.network.addEdge(reader, article)

        self.runRecommendation(readers, plike)
        if self.popular:
        # On every third iteration, "show" the readers the top 5 most popular articles
            if iterations % 3 == 0:
                articleDeg = evaluation.getArticleDegreeDistribution(self.network, 'alive')
                sortedDeg = sorted(articleDeg, key=lambda x: x[1], reverse=True)
                topFive = sortedDeg[0:5]
                for (aId, _) in topFive:
                    article = self.network.getArticle(aId)
                    for reader in readers:
                        probLike = plike(reader, article)
                        if random.random() < probLike:
                            self.network.addEdge(reader, article)
        if self.shouldHelp0DegreeUsers:
            self.help0DegreeUsers(iterations, article)
        if self.shouldHelp0DegreeArticles:
            self.help0DegreeArticles(iterations, self.network.users.values())
        # if iterations % 5 == 0:
        #     users = self.network.getUsersWithDegree0()
        #     for u in users:
        #         probLike = self.PLike(u, article)
        #         if random.random() < probLike:
        #             self.network.addEdge(u, article)
        self.runAnalysis(iterations)

    def runAnalysis(self, iterations):
        for metric in self.metrics:
            self.histories[metric].append(metric.measure(self.network, iterations))

    def killArticles(self, iterations):
        for article in self.network.articles.itervalues():
            #print article
            if not article.getIsDead() and article.getTimeToLive() < iterations:
                article.setIsDead(True)
    
    def runAllSimulation(self):
        for i in util.visual_xrange(self.num_iterations, use_newlines=False):
            if self.simulation == "simulate":
                self.simulate(i, plike = self.plike)
            elif self.simulation == "triadicClosure":
                self.triadicClosureBasedOnFriends(i, force = self.force, plike = self.plike)
            elif self.simulation == "recc":
                self.recc_system_simulate(i, plike = self.plike)
            elif self.simulation == "random":
                self.randomRandomCompleteTriangles(i, plike = self.plike)
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
        >>> experiment.runExperiment(all_analyses=False, num_iterations=10, simulation="recc", recommender='RandomRecommender')

    To save you the time of opening a Python console, you can do this from the shell:
        #> python
    """
    exp = Experiment(*args, **kwargs)
    exp.runAllSimulation()
    exp.saveResults()

    # Save parameters
    with open(out_path('parameters.json'), 'w') as fp:
        json.dump(kwargs, fp)


if __name__ == "__main__":
    runExperiment()
    # http://www.slideshare.net/koeverstrep/tutorial-bpocf
