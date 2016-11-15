import snap
import math
#import scipy
import util
from article import Article
from user import User
import random

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
        self.userList = []
        self.articleList = []
        self.friendGraph = snap.LoadEdgeList(snap.PUNGraph, "stackoverflow-Java.txt", 0, 1)
        #snap.TUNGraph.New()
        self.userArticleGraph = snap.TUNGraph.New()
        self.articleIdCounter = self.largestNodeId(self.friendGraph) + 1
        self.initializeUsers()

    def addArticle(self, article):
        article.setArticleId(self.articleIdCounter)
        self.articleIdCounter = self.articleIdCounter + 1
        self.articleList.append(article)
        self.userArticleGraph.AddNode(article.getArticleId())

    def addEdge(self, user, article):
        uId = user.getUserId()
        aId = article.getArticleId()
        self.userArticleGraph.AddEdge(uId, aId)

    def addUser(self, user):
        self.userList.append(user)
        self.userArticleGraph.AddNode(user.getUserId())

    def getArticlePolticalness(self, articleId):
        return self.getArticle(articleId).getArticlePolticalness()

    def getArticle(self, articleId):
        for article in self.articleList:
            if article.getArticleId() == articleId:
                return article
        return -1

    def getUserIdsWithSpecificPoltiicalness(self, polticalness):
        result = []
        for user in self.userList:
            if user.getPoliticalness() == polticalness:
                result.append(user.getUserId())
        return result

    #intitlaize users independent of their friends
    def initializeUsers(self):
        indexToPoliticalness = {0: -2, 1: -1, 2:0, 3:1, 4:2}
        for node in self.friendGraph.Nodes():
            index = util.generatePoliticalness(self.POLITICALNESS_DISTRIBUTION_FOR_USERS)
            polticalness = indexToPoliticalness[index]
            user = User(polticalness, node.GetId())
            self.addUser(user)



    def getBeta(self, userNodeId):
        SLOPE = .5
        return self.userArticleGraph.GetNI(userNodeId).GetOutDeg() * SLOPE

    def powerLawExponentialCutoff(self, userNodeId, x):
        beta = self.getBeta(userNodeId)

        #uncomment this line when using scipy
        #return beta ** (1-self.ALPHA) * x ** (-self.ALPHA) * math.exp(-beta * x) / scipy.special.gammainc(1-self.ALPHA, beta * self.X_MIN)
        return beta


    def sampleFromPowerLawExponentialCutoff(self, userNodeId):
    	#writing rejciton smapling
    	#g is uniform 0,1
        M = self.powerLawExponentialCutoff(userNodeId, self.X_MIN)
        if M == 0:
            #when have no edges
            return 10 * random.random()
        while True:
            x = random.random()
            r = self.powerLawExponentialCutoff(userNodeId, x) / M
            u = random.random()
            if u <= r:
                return x

    def getNextReaders(self):
        result = []
        for user in self.userList:
            result.append((user, self.sampleFromPowerLawExponentialCutoff(user.getUserId())))
        #Want smallest values
        sortedResults = sorted(result, key=lambda x: x[1])
        readers = []
        for i in range(0, self.NUMBER_OF_READERS):
            readers.append(sortedResults[i][0])
        return readers

