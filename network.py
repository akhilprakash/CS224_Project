import snap
import math
import scipy.special
import util
from user import User
import random
import os


class Network(object):

    ALPHA = .8
    X_MIN = 1
    NUMBER_OF_READERS = 5

    POLITICALNESS_DISTRIBUTION_FOR_USERS = [.1, .3, .2, .3, .1]

    def largestNodeId(self, graph):
        maxNodeId = -1
        for node in graph.Nodes():
            maxNodeId = max(node.GetId(), maxNodeId)
        return maxNodeId

    def __init__(self):
        self.users = {}
        self.articles = {}
        self.friendGraph = snap.LoadEdgeList(snap.PUNGraph, os.path.join("data", "stackoverflow-Java.txt"), 0, 1)
        self.userArticleGraph = snap.TUNGraph.New()
        self.articleIdCounter = self.largestNodeId(self.friendGraph) + 1
        self.initializeUsers()

    def addArticle(self, article):
        article.setArticleId(self.articleIdCounter)
        self.articleIdCounter += 1
        self.articles[article.getArticleId()] = article
        self.userArticleGraph.AddNode(article.getArticleId())

    def addEdge(self, user, article):
        uId = user.getUserId()
        aId = article.getArticleId()
        self.userArticleGraph.AddEdge(uId, aId)

    def addUser(self, user):
        self.users[user.getUserId()] = user
        self.userArticleGraph.AddNode(user.getUserId())

    def getArticlePolticalness(self, articleId):
        return self.getArticle(articleId).getPoliticalness()

    def getArticle(self, articleId):
        return self.articles[articleId]

    def getUserIdsWithSpecificPoliticalness(self, politicalness):
        return [user.getUserId() for user in self.users.itervalues() if user.getPoliticalness() == politicalness]

    def initializeUsers(self):
        # initialize users independent of their friends
        indexToPoliticalness = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        for node in self.friendGraph.Nodes():
            index = util.generatePoliticalness(self.POLITICALNESS_DISTRIBUTION_FOR_USERS)
            polticalness = indexToPoliticalness[index]
            user = User(polticalness, node.GetId())
            self.addUser(user)

    def getBeta(self, userNodeId, slope=.5):
        return self.userArticleGraph.GetNI(userNodeId).GetOutDeg() * slope

    def powerLawExponentialCutoff(self, userNodeId, x):
        beta = self.getBeta(userNodeId)
        return beta
        #return beta ** (1-self.ALPHA) * x ** (-self.ALPHA) * math.exp(-beta * x) / scipy.special.gammainc(1-self.ALPHA, beta * self.X_MIN)

    def sampleFromPowerLawExponentialCutoff(self, userNodeId):
        # Rejection sampling
        # g is uniform 0,1
        M = self.powerLawExponentialCutoff(userNodeId, self.X_MIN)
        if M == 0:
            # when have no edges
            return 10 * random.random()
        while True:
            x = random.random()
            r = self.powerLawExponentialCutoff(userNodeId, x) / M
            u = random.random()
            if u <= r:
                return x

    def getNextReaders(self):
        result = []
        for user in self.users.itervalues():
            result.append((user, self.sampleFromPowerLawExponentialCutoff(user.getUserId())))
        # Want smallest values
        sortedResults = sorted(result, key=lambda x: x[1])
        readers = []
        for i in range(0, self.NUMBER_OF_READERS):
            readers.append(sortedResults[i][0])
        return readers

