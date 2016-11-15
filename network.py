import snap
import math
import scipy
import util
import article
import user


class Network(object):

    ALPHA = .8
    X_MIN = 1
    NUMBER_OF_READERS = 5

    POLITICALNESS_DISTRIBUTION_FOR_USERS = [.1, .3, .2, .3, .1]


    def __init__(self):
        self.userList = []
        self.articleList = []
        self.friendGraph = snap.TUNGraph.New()
        self.userArticleGraph = snap.TUNGraph.New()
        self.articleIdCounter = self.friendGraph.GetNodes() + 1
        initializeUsers(self)

    def addArticle(self, article):
        article.setArticleId(self.articleIdCounter)
        self.articleIdCounter = self.articleIdCounter + 1
        articleList.append(article)
        userArticleGraph.AddNode(article.getArticleId())

    def addEdge(self, user, article):
        uId = user.getUserId()
        aId = article.getArticleId()
        self.userArticleGraph.AddEdge(uId, aId)

    def addUser(self, user):
        userList.append(user)
        userArticleGraph.AddNode(user.getUserId())

    def getArticlePolticalness(self, articleId):
        for article in articleList:
            if article.getArticleId() == articleId:
                return article.getPoliticalness()
        return -1

    def getUserIdsWithSpecificPoltiicalness(self, polticalness):
        result = []
        for user in userList:
            if user.getPoliticalness() == polticalness:
                result.append(user.getUserId())
        return result

    #intitlaize users independent of their friends
    def initializeUsers(self):
        indexToPoliticalness = {0: -2, 1: -1, 2:0, 3:1, 4:2}
        for node in self.friendGraph.Nodes():
            index = util.generatePoliticalness(self, POLITICALNESS_DISTRIBUTION_FOR_USERS)
            polticalness = indexToPoliticalness(index)
            user = User(polticalness, node)
            addUser(self, user)



    def getBeta(self, userNodeId):
        SLOPE = .5
        return userArticleGraph.getOutDeg(userNodeId) * SLOPE

    def powerLawExponentialCutoff(self, userNodeId, x):
        beta = getBeta(userNodeId)

        return beta ** (1-ALPHA) * x ** (-ALPHA) * math.exp(-beta * x) / scipy.special.gammainc(1-ALPHA, beta * X_MIN)


    def sampleFromPowerLawExponentialCutoff(self, userNodeId):
    	#writing rejciton smapling
    	#g is uniform 0,1
        M = powerLawExponentialCutoff(userNodeId, X_MIN)
        while True:
            x = random.random()
            r = powerLawExponentialCutoff(userNodeId, x) / M
            u = random.random()
            if u <= r:
                return x

    def getNextReaders(self):
        result = []
        for user in self.userList:
            result.append(user.getUserId(), sampleFromPowerLawExponentialCutoff(self, user.getUserId()))
        #Want smallest values
        sortedResults = sorted(result, key=itemgetter(2))
        readers = []
        for i in range(0, NUMBER_OF_READERS):
            readers.append(sortedResults[i][0])
        return readers

