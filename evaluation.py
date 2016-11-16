import snap
import network
import random


def getDistribution(network):
    userArticleGraph = network.userArticleGraph
    distribution = {}
    for user in network.users.itervalues():
        nodeUserId = user.getUserId()
        userPolticalness = user.getPoliticalness()
        for article in userArticleGraph.GetNI(nodeUserId).GetOutEdges():
            articlePoliticalness = network.getArticlePolticalness(article)
            if userPolticalness in distribution:
                innerDict = distribution[userPolticalness]
                if articlePoliticalness in innerDict:
                    innerDict[articlePoliticalness] = innerDict[articlePoliticalness] + 1
                else:
                    innerDict[articlePoliticalness] = 1
            else:
                distribution[userPolticalness] = {articlePoliticalness: 1}
    return distribution

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def pathsBetween2Polticalnesses(network, polticalness1=-2,
                                polticalness2=2):
    userArticleGraph = network.userArticleGraph
    negativeTwo = network.getUserIdsWithSpecificPoliticalness(polticalness1)
    posTwo = network.getUserIdsWithSpecificPoliticalness(polticalness2)
    negativeTwo = random.sample(negativeTwo, 10)
    posTwo = random.sample(posTwo, 10)
    distance = []
    for user1 in negativeTwo:
        for user2 in posTwo:
            # 		#figure out why this is not working
            distance.append(
                snap.GetShortPath(userArticleGraph, user1, user2))
    # x = 1
    return mean(distance)

def modularity(network):
    Nodes = snap.TIntV()
    for nodeId in network.userArticleGraph.Nodes():
        Nodes.Add(nodeId)
    return snap.getModularity(network.userArticleGraph, Nodes)

def betweeness(network):

    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(network.userArticleGraph, Nodes, Edges, 1.0)
    for node in Nodes:
        print "node: %d centrality: %f" % (node, Nodes[node])
    # for edge in Edges:
    #   		print "edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge])

    betweenessCentr = []
    for edge in Edges:
        betweenessCentr.append((edge, Edges[edge]))

    return betweenessCentr

def getUserDegreeDistribution(network, polticalness="all"):
    userArticleGraph = network.userArticleGraph
    degree = []
    for user in network.users.itervalues():
        uId = user.getUserId()
        if polticalness == "all" or (
            str(user.getPoliticalness()) == polticalness):
            degree.append(userArticleGraph.GetNI(uId).GetOutDeg())
    return degree

def getArticleDegreeDistribution(network, str):
    userArticleGraph = network.userArticleGraph
    degree = []
    for article in network.articles.itervalues():
        aId = article.getArticleId()
        if str == "all" or (str == "alive" and not article.getIsDead()) or (str == "dead" and article.getIsDead()):
            degree.append((aId, userArticleGraph.GetNI(aId).GetOutDeg()))
    return degree

def getDistributionOfLifeTime(network, iterations):
    lifeTime = []
    for article in network.articles.itervalues():
        if not article.getIsDead():
            lifeTime.append(article.getTimeToLive() - iterations)
    return lifeTime

# triangles
def clusterOneNode(node, graph):
    degree = node.GetOutDeg()
    if degree < 2:
        return 0
    neighborsOfNode = node.GetOutEdges()
    counter = 0
    for id in neighborsOfNode:
        for k in node.GetOutEdges():
            if graph.IsEdge(k, id):
                counter = counter + 1
    counter = counter / 2
    return (2.0 * counter) / (degree * (degree - 1))

def clustersForUsers(polticalness="all"):
    userArticleGraph = network.userArticleGraph
    cluster = []
    for user in network.users.itervalues():
        if polticalness == "all" or str(
                user.getPoliticalness()) == polticalness:
            result = clusterOneNode(
                userArticleGraph.GetNI(user.getUserId()), userArticleGraph)
            cluster.append(result)
    return cluster
