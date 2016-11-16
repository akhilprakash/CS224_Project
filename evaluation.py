"""
All metrics should be a subclass of Metric, and at least implement the measure
method.
"""
import snap
import random
import util
from util import print_error, out_path
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


class Metric(object):
    @property
    def name(self):
        return self.__class__.__name__

    def measure(self, network, iterations):
        """Given a Network, return a metric of any type."""
        raise NotImplementedError

    def plot(self, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        print_error("No plotter for %s" % self.name)

    def save(self, history):
        """
        Save history to a file.
        """
        print_error("No saver for %s" % self.name)

    # Helpful Stuff #
    def __hash__(self):
        params = vars(self)
        keys = params.keys()
        keys.sort()
        return hash((self.name, tuple(params[k] for k in keys)))

    def __eq__(self, other):
        return self.name == self.name and vars(self) == vars(other)

    def __str__(self):
        params = ', '.join(['%s=%s' % (k, v) for k, v in vars(self)])
        return '%s(%s)' % (self.name, params)

    def __repr__(self):
        return self.__str__()


class ReadingDistribution(Metric):
    """
    Distribution of article political leaning given user political leaning.
    """
    def measure(self, network, iterations):
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


class PathsBetweenPoliticalnesses(Metric):
    def __init__(self, politicalness1=-2, politicalness2=2):
        self.politicalness1 = politicalness1
        self.politicalness2 = politicalness2

    def measure(self, network, iterations):
        userArticleGraph = network.userArticleGraph
        negativeTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness1)
        posTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness2)
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

    def plot(self, history):
        plt.figure()
        plt.plot(history)
        plt.savefig(out_path(self.name))


class Modularity(Metric):
    def measure(self, network, iterations):
        Nodes = snap.TIntV()
        for nodeId in network.userArticleGraph.Nodes():
            Nodes.Add(nodeId)
        return snap.getModularity(network.userArticleGraph, Nodes)


class Betweenness(Metric):
    def measure(self, network, iterations):
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


class UserDegreeDistribution(Metric):
    def __init__(self, politicalness="all"):
        self.politicalness = politicalness

    def measure(self, network, iterations):
        userArticleGraph = network.userArticleGraph
        degree = []
        for user in network.users.itervalues():
            uId = user.getUserId()
            if self.politicalness == "all" or (str(user.getPoliticalness()) == self.politicalness):
                degree.append(userArticleGraph.GetNI(uId).GetOutDeg())
        return degree

    def plot(self, history):
        util.writeCSV(out_path("userDegree"), self.history)


def getArticleDegreeDistribution(network, article_type):
    userArticleGraph = network.userArticleGraph
    degree = []
    for article in network.articles.itervalues():
        aId = article.getArticleId()
        if article_type == "all" or \
                (article_type == "alive" and not article.getIsDead()) or \
                (article_type == "dead" and article.getIsDead()):
            degree.append((aId, userArticleGraph.GetNI(aId).GetOutDeg()))
    return degree


class ArticleDegreeDistribution(Metric):
    def __init__(self, article_type):
        self.article_type = article_type

    def measure(self, network, iterations):
        return map(lambda x: x[1], getArticleDegreeDistribution(network, self.article_type))

    def plot(self, history):
        # This is for Akhil's R plots
        if self.article_type == 'all':
            util.writeCSV(out_path("articleDegree"), history)
        elif self.article_type == 'alive':
            print_error('skipping CSV for alive article degree dist')
        elif self.article_type == 'dead':
            util.writeCSV(out_path("deadArticle"), history)


class DistributionOfLifeTime(Metric):
    def measure(self, network, iterations):
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


def clustersForUsers(network, polticalness="all"):
    userArticleGraph = network.userArticleGraph
    cluster = []
    for user in network.users.itervalues():
        if polticalness == "all" or str(
                user.getPoliticalness()) == polticalness:
            result = clusterOneNode(
                userArticleGraph.GetNI(user.getUserId()), userArticleGraph)
            cluster.append(result)
    return cluster
