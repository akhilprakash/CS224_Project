import itertools
import pdb
import random
from sets import Set
import snap
from scipy.sparse import csc_matrix

import util
from user import User
import networkx as nx

def getMax(NIdToDistH):
    nodeId = -1
    longestPath = -1
    for item in NIdToDistH:
        if NIdToDistH[item] > longestPath:
            longestPath = NIdToDistH[item]
            nodeId = item
    return (nodeId, longestPath)


class Network(object):
    ALPHA = .8
    X_MIN = 1

    # TODO: is this based on data?
    POLITICALNESS_DISTRIBUTION_FOR_USERS = [.1, .3, .2, .3, .1]

    def largestNodeId(self, graph):
        maxNodeId = -1
        for node in graph.Nodes():
            maxNodeId = max(node.GetId(), maxNodeId)
        return maxNodeId

    def __init__(self, friendGraphFile, initMethod):
        self.users = {}
        self.articles = {}
        if friendGraphFile[-3:] == "csv":
            self.friendGraph = snap.LoadEdgeList(snap.PUNGraph, friendGraphFile, 0, 1, ",")    
        else:
            self.friendGraph = snap.LoadEdgeList(snap.PUNGraph, friendGraphFile, 0, 1, "\t")
        print self.friendGraph.GetNodes()
        self.userArticleGraph = snap.TUNGraph.New()
        self.articleIdCounter = self.largestNodeId(self.friendGraph) + 1
        if friendGraphFile[-3:] == "csv":
            self.userArticleFriendGraph = snap.LoadEdgeList(snap.PUNGraph, friendGraphFile, 0, 1, ",")
        else:
            self.userArticleFriendGraph = snap.LoadEdgeList(snap.PUNGraph, friendGraphFile, 0, 1, "\t")
        if initMethod == "propagation":
            self.initializeUsersBasedOn2Neg2()
        elif initMethod == "random":
            self.initializeUsers()
        elif initMethod == "friends":
            self.initializeUsersAccordingToFriends()
        else:
            raise Exception("initMethod must be propagation, random, or friends")
        print "done initializing"

    def spreadPoliticalness(self, nodeId, depth):
        political = self.users[nodeId].getPoliticalness()
        retVal = []
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
            retVal.append(newNode)
        return retVal

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
        components = snap.TCnComV()
        snap.GetWccs(self.friendGraph, components)
        print ("Number of weakly connected components: " + str(len(components)))
        for component in components:
            nodesInComponent = []
            for node in self.friendGraph.Nodes():
                if component.IsNIdIn(node.GetId()):
                    nodesInComponent.append(node.GetId())
            counter = 0
            sourceId = -1
            destId = -1
            longestPath = -1
            for source in nodesInComponent:
                if counter > 600:
                    print "broke"
                    print longestPath
                    break
                NIdToDistH = snap.TIntH()
                #print source
                snap.GetShortPath(self.friendGraph, source, NIdToDistH)
                result = getMax(NIdToDistH)
                if result[1] > longestPath:
                    sourceId = source
                    destId = result[0]
                counter = counter + 1

            #arbitrarly assign source ot -2 and dest to 2
            self.users[sourceId].setPoliticalness(-2)
            self.users[destId].setPoliticalness(2)
            fromSourceQueue = self.spreadPoliticalness(sourceId, 0)
            fromDestQueue = self.spreadPoliticalness(destId, 0)
            depth = 1
            visited = Set()
            while self.areUsersUnassigned():
                #print "Source queue length" + str(len(fromSourceQueue))
                #print "Dest queue length" + str(len(fromDestQueue))
                if len(fromSourceQueue) != 0:
                    source = fromSourceQueue.pop(0)
                    if not source in visited:
                        visited.add(source)
                        fromSourceQueue.extend(self.spreadPoliticalness(source, depth))
                if len(fromDestQueue) != 0:
                    dest = fromDestQueue.pop(0)
                    if not dest in visited:
                        visited.add(dest)
                        fromDestQueue.extend(self.spreadPoliticalness(dest, depth))
                if len(fromSourceQueue) == 0 and len(fromDestQueue) == 0 and self.areUsersUnassigned():
                    break
                depth = depth + 1
        if self.areUsersUnassigned():
            raise Exception("Did not assign all suers polticalness")
        self.getPoliticalAllUsers()

    '''
    @return: The first thing in the tuple is the networkX version of the user user graph
             The second thing in the tuple is the snap version of the user user graph
             The third thing in the tuple is the edge to weight dictionary
    '''
    def createUserUserGraph(self):
        G = nx.Graph()
        edgeToWeightDict = util.PairsDict()
        userUserGraph = snap.TUNGraph.New()
        for uId in self.users.keys():
            userUserGraph.AddNode(uId)
        for uId1 in self.users.keys():
            for uId2 in self.users.keys():
                Nbrs = snap.TIntV()
                snap.GetCmnNbrs(self.userArticleGraph, uId1, uId2, Nbrs)
                if self.userArticleGraph.GetNI(uId1).GetOutDeg() + self.userArticleGraph.GetNI(uId2).GetOutDeg() == 0:
                    weight = 0
                else:
                    weight = float(len(Nbrs)) / (self.userArticleGraph.GetNI(uId1).GetOutDeg() + self.userArticleGraph.GetNI(uId2).GetOutDeg())
                    G.add_edge(uId1, uId2, weight = weight)
                    edgeToWeightDict[(uId1, uId2)] = weight
                    userUserGraph.AddEdge(uId1, uId2)
        
        return (G, userUserGraph, edgeToWeightDict)


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

    def removeArticle(self, articleId):
        del self.articles[articleId]
        self.userArticleGraph.DelNode(articleId)
        self.userArticleFriendGraph.DelNode(articleId)

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
                print "finished a user"
        self.getPoliticalAllUsers()

    def getBeta(self, userNodeId, slope=.5):
        return self.userArticleGraph.GetNI(userNodeId).GetOutDeg() * slope

    def powerLawExponentialCutoff(self, userNodeId, x):
        beta = self.getBeta(userNodeId)
        return beta

    # TODO: verify that this works and is efficient enough?
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

    def getNextReaders(self, N):
        result = []
        for user in self.users.itervalues():
            result.append((user, self.sampleFromPowerLawExponentialCutoff(user.getUserId())))
        # Want smallest values
        sortedResults = sorted(result, key=lambda x: x[1])
        readers = []
        for i in range(0, min(N, len(sortedResults))):
            readers.append(sortedResults[i][0])
        return readers

    def getUser(self, userId):
        return self.users[userId]

    def getLikers(self, articleId):
        """Iterator over users that has liked the given article."""
        return self.userArticleGraph.GetNI(articleId).GetOutEdges()

    def reapArticles(self, t):
        # Kill articles that are past their time
        for article in self.articles.values():
            if article.timeToLive < t:
                article.isDead = True
                numLikes = self.userArticleGraph.GetNI(article.articleId).GetDeg()

                # Completely remove articles that never got likes
                if numLikes == 0:
                    self.removeArticle(article.articleId)

    def removeUnlikedArticles(self):
        for article in self.articles.values():
            numLikes = self.userArticleGraph.GetNI(article.articleId).GetDeg()
            if numLikes == 0:
                self.removeArticle(article.articleId)

    def articlesLikedByUser(self, userId):
        """Iterator over the articles liked by the given user."""
        return (self.articles[i] for i in self.userArticleGraph.GetNI(userId).GetOutEdges())

    def getArticles(self):
        """Iterator over articles"""
        return self.articles.itervalues()

    def getLiveArticles(self):
        """Iterator over not-dead articles"""
        return (article for article in self.articles.itervalues() if not article.isDead)

    def candidateArticlesForUser(self, userId):
        """Iterator over alive articles that are not yet liked by the given user."""
        return (
            article
            for article in self.articles.itervalues()
            if not self.userArticleGraph.IsEdge(article.articleId, userId)
            and not article.isDead
        )

    def getUserUserGraph(self):
        """
        Weights are defined by the number of common articles liked.
        The more that two people have read, the more common basis of
        understanding they have.
        """
        weights = util.PairsDict()
        userUserGraph = snap.TUNGraph.New()
        for userId in self.users:
            userUserGraph.AddNode(userId)
        for articleId in self.articles:
            for userA, userB in itertools.combinations(self.getLikers(articleId), 2):
                if not userUserGraph.IsEdge(userA, userB):
                    weights[userA, userB] = 1
                    userUserGraph.AddEdge(userA, userB)
                else:
                    weights[userA, userB] += 1
        return userUserGraph, weights

    def getUserUserGraphMatrix(self):
        """
        Weights are defined by the number of common articles liked.
        The more that two people have read, the more common basis of
        understanding they have.
        """
        weights = util.PairsDict()
        for articleId in self.articles:
            for userA, userB in itertools.combinations(self.getLikers(articleId), 2):
                if (userA, userB) not in weights:
                    weights[userA, userB] = 1
                else:
                    weights[userA, userB] += 1
        i = []
        j = []
        weightsList = []
        for (userA, userB), num_common in weights.iteritems():
            i.append(userA)
            j.append(userB)
            weightsList.append(num_common)
            i.append(userB)
            j.append(userA)
            weightsList.append(num_common)

        return csc_matrix((weightsList, (i, j)))
