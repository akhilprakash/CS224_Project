import snap
import math
import scipy.special
import util
from user import User
import random
import os
import collections
import pdb
import numpy as np


def getMax(NIdToDistH):
    nodeId = -1
    longestPath = -1
    for item in NIdToDistH:
        if NIdToDistH[item] > longestPath:
            longestPath = NIdToDistH[item]
            nodeId = item
    return (item, longestPath)


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

    def __init__(self, initialize):
        self.users = {}
        self.articles = {}
        self.friendGraph = snap.LoadEdgeList(snap.PUNGraph, os.path.join("data", "zacharys.csv"), 0, 1, ",")
        #snap.LoadEdgeList(snap.PUNGraph, os.path.join("data", "stackoverflow-Java-small.txt"), 0, 1)
        self.userArticleGraph = snap.TUNGraph.New()
        self.articleIdCounter = self.largestNodeId(self.friendGraph) + 1
        self.userArticleFriendGraph = snap.LoadEdgeList(snap.PUNGraph, os.path.join("data", "zacharys.csv"), 0, 1, ",")
        #snap.LoadEdgeList(snap.PUNGraph, os.path.join("data", "stackoverflow-Java-small.txt"), 0, 1)
        if initialize == "1":
            self.initializeUsersBasedOn2Neg2()
        elif initialize == "2":
            self.initializeUsers()
        elif initialize == "3":
            self.initializeUsersAccordingToFriends()

    def spreadPoliticalness(self, nodeId, depth):
        political = self.users[nodeId].getPoliticalness()
        for newNode in self.friendGraph.GetNI(nodeId).GetOutEdges():
            #either pass on political, political - 1, oor poltial + 1
            weights = [.3 + depth, .5 + depth, .3 + depth]
            idx = util.weighted_choice(weights)
            if political == 2 or political == -2:
                weights = [.5 + depth, .5 + depth]
                idx = util.weighted_choice(weights)
                if idx == 0:
                    self.users[newNode].setPoliticalness(political)
                else:
                    self.users[newNode].setPoliticalness(political - political / 2)
            else:
                if idx == 0:
                    self.users[newNode].setPoliticalness(political-1)
                elif idx == 1:
                    self.users[newNode].setPoliticalness(political)
                else:
                    self.users[newNode].setPoliticalness(political+1)
        return [v for v in self.friendGraph.GetNI(nodeId).GetOutEdges()]

    def areUsersUnassigned(self):
        for user in self.users.itervalues():
            if user.getPoliticalness() == "NA":
                return True
        return False

    #find the two nodes furthest from each other in the friends graph
    #assign them -2 and 2
    #then porpaorpagte values
    def initializeUsersBasedOn2Neg2(self):
        for node in self.friendGraph.Nodes():
            user = User("NA", node.GetId())
            self.addUser(user)

        sourceId = -1
        destId = -1
        longestPath = -1
        for source in self.friendGraph.Nodes():
            NIdToDistH = snap.TIntH()
            snap.GetShortPath(self.friendGraph, source.GetId(), NIdToDistH)
            result = getMax(NIdToDistH)
            if result[1] > longestPath:
                sourceId = source.GetId()
                destId = result[0]

        #arbitrarly assign source ot -2 and dest to 2
        self.users[sourceId].setPoliticalness(-2)
        self.users[destId].setPoliticalness(2)
        fromSourceQueue = self.spreadPoliticalness(sourceId, 0)
        fromDestQueue = self.spreadPoliticalness(destId, 0)
        depth = 1
        while self.areUsersUnassigned():
            source = fromSourceQueue.pop(0)
            fromSourceQueue.extend(self.spreadPoliticalness(source, depth))
            dest = fromDestQueue.pop(0)
            fromDestQueue.extend(self.spreadPoliticalness(dest, depth))
            depth = depth + 1
        self.getPoliticalAllUsers()

    def calcAdjacencyMatrix(self, graph):
        counter = 0
        uIdOrAIdToMatrix = {}
        for uId, user in self.users.items():
            uIdOrAIdToMatrix[uId] = counter
            counter = counter + 1
        for aId, article in self.articles.items():
            uIdOrAIdToMatrix[aId] = counter
            counter = counter + 1
        matrix = [[0 for _ in range(0, counter)] for _ in range(0, counter)]
        for edges in graph.Edges():
            src = edges.GetSrcNId()
            dest = edges.GetDstNId()
            matrix[uIdOrAIdToMatrix[src]][uIdOrAIdToMatrix[dest]] = 1
            matrix[uIdOrAIdToMatrix[dest]][uIdOrAIdToMatrix[src]] = 1
        return (matrix, uIdOrAIdToMatrix)

    def addArticle(self, article):
        article.setArticleId(self.articleIdCounter)
        self.articleIdCounter += 1
        self.articles[article.getArticleId()] = article
        self.userArticleGraph.AddNode(article.getArticleId())
        self.userArticleFriendGraph.AddNode(article.getArticleId())

    def addEdge(self, user, article):
        uId = user.getUserId()
        aId = article.getArticleId()
        self.userArticleGraph.AddEdge(uId, aId)
        self.userArticleFriendGraph.AddEdge(uId, aId)

    def addUser(self, user):
        self.users[user.getUserId()] = user
        self.userArticleGraph.AddNode(user.getUserId())

    def getArticle(self, articleId):
        return self.articles[articleId]

    def getUserIdsWithSpecificPoliticalness(self, politicalness):
        return [user.getUserId() for user in self.users.itervalues() if user.getPoliticalness() == politicalness]

    def initializeUsers(self):
        # initialize users independent of their friends
        indexToPoliticalness = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}
        for node in self.friendGraph.Nodes():
            index = util.weighted_choice(self.POLITICALNESS_DISTRIBUTION_FOR_USERS)
            politicalness = indexToPoliticalness[index]
            user = User(politicalness, node.GetId())
            self.addUser(user)

    def getPoliticalAllUsers(self):
        political = []
        for user in self.users.values():
            political.append(user.getPoliticalness())
        print political
        return political

    def initializeUsersAccordingToFriends(self):
        #intilaize users randomly
        self.initializeUsers()
        self.getPoliticalAllUsers()
        NUM_ITERATIONS = 1
        for _ in range(0, NUM_ITERATIONS):
            keys = self.users.keys()
            random.shuffle(keys)
            for userId in keys:
                potlicalnessOfFriends = [0 for _ in range(-2, 3)]
                for friend in self.friendGraph.GetNI(userId).GetOutEdges():
                    userFriend = self.getUser(friend)
                    potlicalnessOfFriends[userFriend.getPoliticalness()+2] = potlicalnessOfFriends[userFriend.getPoliticalness()+2] + 1
                user = self.getUser(userId)
                idx = util.weighted_choice(potlicalnessOfFriends)
                if potlicalnessOfFriends[idx] == 0:
                    pdb.set_trace()
                user.setPoliticalness(idx -2)
        self.getPoliticalAllUsers()

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

    def getArticles(self):
        """Iterator over articles that aren't dead."""
        return (article for article in self.articles.itervalues() if not article.isDead)

    def getRandomArticles(self, N):
        return random.sample(list(self.getArticles()), N)

    def getUser(self, userId):
        return self.users[userId]

    def articlesLikedByUser(self, userId):
        """Iterator over the articles liked by the given user."""
        return (
            self.articles[i]
            for i in self.userArticleGraph.GetNI(userId).GetOutEdges()
            if not self.articles[i].isDead
        )
