"""
All metrics should be a subclass of Metric, and at least implement the measure
method.
"""
import snap
import network
import random
import random
import util
from util import print_error, out_path
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
import scipy
import numpy
import pdb

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


class Metric(object):
    @property
    def name(self):
        return self.__class__.__name__

    @property
    def safe_name(self):
        params = '_'.join(['%s_%s' % (k, v) for k, v in vars(self).iteritems()])
        return '%s_%s' % (self.name, params)

    def measure(self, network, iterations):
        """Given a Network, return a metric of any type."""
        raise NotImplementedError

    def plot(self, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        print_error("No plotter for %s" % self)

    def save(self, history):
        """
        Save history to a file.
        """
        print_error("No saver for %s" % self)

    # Helpful Stuff #
    def __hash__(self):
        params = vars(self)
        keys = params.keys()
        keys.sort()
        return hash((self.name, tuple(params[k] for k in keys)))

    def __eq__(self, other):
        return self.name == self.name and vars(self) == vars(other)

    def __str__(self):
        params = ', '.join(['%s=%r' % (k, v) for k, v in vars(self).iteritems()])
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
                articlePoliticalness = network.getArticle(article).getPoliticalness()
                if userPolticalness in distribution:
                    innerDict = distribution[userPolticalness]
                    if articlePoliticalness in innerDict:
                        innerDict[articlePoliticalness] = innerDict[articlePoliticalness] + 1
                    else:
                        innerDict[articlePoliticalness] = 1
                else:
                    distribution[userPolticalness] = {articlePoliticalness: 1}
        return distribution

    def plot(self, history):
        last = history[-1]
        for key, value in last.items():
            #value is a dictionary
            keys = []
            vals = []
            for k1, v1 in value.items():
                keys.append(k1)
                vals.append(v1)
            plt.figure()
            plt.bar(keys, vals, color = "blue")
            plt.xlabel("Article Politicalness")
            plt.ylabel("Frequency")
            plt.title("Which Articles do Users with polticalness " + str(key) + " Read")
            #make this a mosaic plot later
            plt.savefig(out_path(self.safe_name + "key=" + str(key) + ".png"))
            plt.close()

    def save(self, history):
        util.writeCSV(out_path("readingDistribution"), history)

class PathsBetweenPoliticalnesses(Metric):
    def __init__(self, politicalness1=-2, politicalness2=2):
        self.politicalness1 = politicalness1
        self.politicalness2 = politicalness2

    def measure(self, network, iterations):
        userArticleGraph = network.userArticleGraph
        negativeTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness1)
        posTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness2)
        negativeTwo = random.sample(negativeTwo, min(20, len(negativeTwo)))
        posTwo = random.sample(posTwo, min(20, len(posTwo)))
        distance = []
        for user1 in negativeTwo:
            for user2 in posTwo:
                path = snap.GetShortPath(userArticleGraph, user1, user2)
                if path >= 0:
                    distance.append(path)
        return mean(distance)

    def plot(self, history):
        plt.figure()
        plt.plot(history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Average Distance Between Polticalness")
        plt.title("Average Distance Between " + str(self.politicalness1) + " and " + str(self.politicalness2))
        plt.savefig(out_path(self.safe_name + '.png'))

    def save(self, history):
        util.writeCSV(out_path("pathsbetweenpolticalness" + str(self.politicalness1) + str(self.politicalness2)), history) 

class Modularity(Metric):
    def measure(self, network, iterations):
        Nodes = snap.TIntV()
        for ni in network.userArticleGraph.Nodes():
            Nodes.Add(ni.GetId())
        print snap.GetModularity(network.userArticleGraph, Nodes)
        return snap.GetModularity(network.userArticleGraph, Nodes)

    def plot(self, history):
        plt.figure()
        plt.plot(history)
        plt.savefig(out_path(self.safe_name + '.png'))
        plt.close()


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
        # Just plot the dist for the last iteration
        last = history[-1]
        plt.figure()
        plt.hist(last)
        plt.xlabel("User Degree")
        plt.ylabel("Frequency")
        plt.title("Histogram of User Degree")
        plt.savefig(out_path(self.safe_name + '.png'))
        plt.close()

    def save(self, history):
        util.writeCSV(out_path("userDegree"), history)


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
        for i,h in enumerate(history):
            plt.figure()
            plt.hist(h)
            plt.xlabel("Aricle Degree")
            plt.ylabel("Frequency")
            plt.title("Histogram of Article Degree")
            plt.savefig(out_path(self.safe_name + self.article_type + "time=" + str(i) + '.png'))
            plt.close()

    def save(self, history):
        # This is for Akhil's R plots
        if self.article_type == 'all':
            util.writeCSV(out_path("articleDegree"), history)
        elif self.article_type == 'alive':
            util.writeCSV(out_path("aliveAricleDegree"), history)
        elif self.article_type == 'dead':
            util.writeCSV(out_path("deadArticle"), history)


class DistributionOfLifeTime(Metric):

    def __init__(self, article_type):
        self.article_type = article_type

    def measure(self, network, iterations):
        lifeTime = []
        for article in network.articles.itervalues():
            if self.article_type == "alive" and not article.getIsDead():
                lifeTime.append(article.getTimeToLive() - iterations)
            if self.article_type == "dead" and article.getIsDead():
                lifeTime.append(article.getTimeToLive() - iterations)
        return lifeTime

    def plot(self, history):
        for i,h in enumerate(history):
            plt.figure()
            plt.hist(h)
            plt.xlabel("Aricle Lifetime")
            plt.ylabel("Frequency")
            plt.title("Histogram of Article Lifetime")
            plt.savefig(out_path(self.safe_name + self.article_type + "time=" + str(i) + '.png'))
            plt.close()

    def save(self, history):
        util.writeCSV(out_path("article lifetime disitburtion" + self.article_type), history)



class AliveArticles(Metric):
    def measure(self, network, iterations):
        counterAlive = 0
        counterDead = 0
        for article in network.articles.itervalues():
            if not article.getIsDead():
                counterAlive = counterAlive + 1
            else:
                counterDead = counterDead + 1
        return counterAlive

    def plot(self, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        numIterations = len(history)
        plt.figure()
        plt.plot(range(0, numIterations), history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Number of Alive Articles")
        plt.title("Number of Alive Articles vs. Number of Iterations")
        plt.savefig(out_path(self.safe_name + '.png'))
        plt.close()

    def save(self, history):
        """
        Save history to a file.
        """
        util.writeCSV(out_path("numberAliveArticles"), history)

class OverallClustering(Metric):
    def measure(self, network, iterations):
        return snap.GetClustCf(network.userArticleGraph)

    def plot(self, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        plt.figure()
        plt.plot(history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Clustering Coefficient")
        plt.title("Clustering Coefficient vs. Number of Iterations")
        plt.savefig(out_path(self.safe_name + '.png'))

    def save(self, history):
        """
        Save history to a file.
        """
        util.writeCSV(out_path("OverallClustering"), history)


class DeadArticles(Metric):
    def measure(self, network, iterations):
        counterDead = 0
        for article in network.articles.itervalues():
            if article.getIsDead():
                counterDead = counterDead + 1
        return counterDead

    def plot(self, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        numIterations = len(history)
        plt.figure()
        plt.plot(range(0, numIterations), history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Number of Dead Articles")
        plt.title("Number of Dead Articles vs. Number of Iterations")
        plt.savefig(out_path(self.safe_name + '.png'))
        plt.close()

    def save(self, history):
        """
        Save history to a file.
        """
        util.writeCSV(out_path("numberDeadArticles"), history)

def printGraph(graph):
    for EI in graph.Edges():
        print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())

class ClusterPolticalness(Metric):

    def __init__(self, polticalness):
        self.polticalness = polticalness

    # triangles
    def clusterOneNode(self, node, graph):
        #printGraph(graph)
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

    def measure(self, network, iterations):
        userArticleGraph = network.userArticleGraph
        cluster = []
        for user in network.users.itervalues():
            #if iterations > 35 and self.polticalness == "all":
                #pdb.set_trace()
            if self.polticalness == "all" or str(
                    user.getPoliticalness()) == self.polticalness:
                result = self.clusterOneNode(
                    userArticleGraph.GetNI(user.getUserId()), userArticleGraph)
                cluster.append(result)
        return mean(cluster)

    def plot(self, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        numIterations = len(history)
        print history
        plt.figure()
        plt.plot(range(0, numIterations), history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Clustering Coefficient")
        plt.title("Clustering Coefficient for polticalness " + str(self.polticalness) + "\n vs. Number of Itertions")
        plt.savefig(out_path(self.safe_name + "polticialness" + str(self.polticalness) + '.png'))
        plt.close()

    def save(self, history):
        """
        Save history to a file.
        """
        util.writeCSV(out_path("clusterPolticalness" + "_polticalness=" + self.polticalness), history)

class LargestConnectedComponent(Metric):

    def measure(self, network, iterations):
        Components = snap.TCnComV()
        snap.GetWccs(network.userArticleGraph, Components)
        numComponents = []
        for CnCom in Components:
            numComponents.append(CnCom.Len())
        return numComponents

    def plot(self, history):
        for i,elem in enumerate(history):
            plt.figure()
            plt.bar(range(len(elem)), elem)
            plt.savefig(out_path(self.safe_name + "connected_compoenents_" + str(i) + '.png'))
            plt.close()
        largestComponent = map(max, history)
        plt.figure()
        plt.plot(largestComponent)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Size of Largest Connected Component")
        plt.title("Size of Largest Connected Component vs. Number of Iterations")
        plt.savefig(out_path(self.safe_name + "largest_component" + ".png"))
        plt.clf()
        plt.close()
        numComponents = map(len, history)
        plt.figure()
        plt.plot(numComponents)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Number of Components")
        plt.title("Number of Components vs. Number of Iterations")
        plt.savefig(out_path(self.safe_name + "number_components" + ".png"))
        plt.clf()
        plt.close()

    def save(self, history):
        largestComponent = map(max, history)
        util.writeCSV(out_path("largestCompeonent"), history)
        numComponents = map(len, history)
        util.writeCSV(out_path("numberCompeonent"), history)


class EigenVectors(Metric):

    def measure(self, network, iterations):
        EigvV =  snap.TFltV()
        snap.GetEigVec(network.userArticleGraph, EigvV)
        result = []
        for Val in EigvV:
            result.append(Val)
        return sorted(result)

    def plot(self, history):
        last = history[-1]
        plt.figure()
        plt.plot(last)
        plt.xlabel("Rank of Eigenvector")
        plt.ylabel("Values of Eigenvector")
        plt.title("First Eigenvector")
        plt.savefig(out_path(self.safe_name + ".png"))
        plt.close()

def getEigenVectorEigenValue(network):
    counter = 0
    uIdOrAIdToMatrix = {}
    for uId, user in network.users.items():
        uIdOrAIdToMatrix[uId] = counter
        counter = counter + 1
    for aId, article in network.articles.items():
        uIdOrAIdToMatrix[aId] = counter
        counter = counter + 1
    matrix = [[0 for _ in range(0, counter)] for _ in range(0, counter)]
    for edges in network.userArticleGraph.Edges():
        src = edges.GetSrcNId()
        dest = edges.GetDstNId()
        matrix[uIdOrAIdToMatrix[src]][uIdOrAIdToMatrix[dest]] = 1
        matrix[uIdOrAIdToMatrix[dest]][uIdOrAIdToMatrix[src]] = 1
    #print matrix
    #print len(matrix)
    #print len(matrix[0])
    laplacian = scipy.sparse.csgraph.laplacian(numpy.array(matrix))
    eigenvalue, eigenvector = numpy.linalg.eig(laplacian)
    #print eigenvalue
    #print eigenvector
    #result = [x for (y,x) in sorted(zip(eigenvalue,eigenvector))]
    eigenvalueIdx = eigenvalue.argsort()
    result = eigenvector[:, eigenvalueIdx]
    return (result, uIdOrAIdToMatrix, matrix)

class MoreEigenVectors(Metric):

    def measure(self, network, iterations):
        # counter = 0
        # uIdOrAIdToMatrix = {}
        # for uId, user in network.users.items():
        #     uIdOrAIdToMatrix[uId] = counter
        #     counter = counter + 1
        # for aId, article in network.articles.items():
        #     uIdOrAIdToMatrix[aId] = counter
        #     counter = counter + 1
        # matrix = [[0 for _ in range(0, counter)] for _ in range(0, counter)]
        # for edges in network.userArticleGraph.Edges():
        #     src = edges.GetSrcNId()
        #     dest = edges.GetDstNId()
        #     matrix[uIdOrAIdToMatrix[src]][uIdOrAIdToMatrix[dest]] = 1
        #     matrix[uIdOrAIdToMatrix[dest]][uIdOrAIdToMatrix[src]] = 1
        # laplacian = scipy.sparse.csgraph.laplacian(matrix)
        # eigenvalue, eigenvector = numpy.linalg.eig(laplacian)
        # result = [x for (y,x) in sorted(zip(eigenvalue,eigenvector))]
        result = getEigenVectorEigenValue(network)
        eigenvector = result[0]
        return eigenvector[:,1]

    def plot(self, history):
        for i,eigenvector in enumerate(history):
            sortedEigenvector = sorted(eigenvector)
            plt.figure()
            plt.plot(sortedEigenvector)
            plt.xlabel("Rank of Eigenvector")
            plt.ylabel("Values of Eigenvector")
            plt.title("Second Eigenvector")
            plt.savefig(out_path(self.safe_name + "time=" + str(i) + ".png"))
            plt.close()

#number of common articles between users
class CommonArticles(Metric):

    def __init__(self, politicalness1=-2, politicalness2=2):
        self.politicalness1 = politicalness1
        self.politicalness2 = politicalness2

    def measure(self, network, iterations):
        userArticleGraph = network.userArticleGraph
        negativeTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness1)
        posTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness2)
        negativeTwo = random.sample(negativeTwo, min(20, len(negativeTwo)))
        posTwo = random.sample(posTwo, min(20, len(posTwo)))
        commonNeighs = []
        
        for s in posTwo:
            for v in negativeTwo:
                Nbrs = snap.TIntV()
                snap.GetCmnNbrs(userArticleGraph, s, v, Nbrs)
                commonNeighs.append(len(Nbrs))
        return mean(commonNeighs)

    def plot(self, history):
        plt.figure()
        plt.plot(history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Common Neighbors")
        plt.title("Common Neighbors between " + str(self.politicalness1) + " and " + str(self.politicalness2))
        plt.savefig(out_path(self.safe_name + "polticalness=" + str(self.politicalness1) + " and " + str(self.politicalness2) + ".png"))
        plt.close()


    def save(self, history):
        util.writeCSV(out_path("CommonArticles_" + "polticalness=" + str(self.politicalness1) + " and " + str(self.politicalness2)), history)

class VisualizeGraph(Metric):

    def measure(self, network, iterations):
        eigenvector, dictionary, matrix = getEigenVectorEigenValue(network)
        
        twoEigenVectors = eigenvector[1:3,:]
        #pdb.set_trace()
        #print twoEigenVectors
        self.network = network
        return (twoEigenVectors, dictionary, matrix)

    def plot(self, history):
        counter = 0
        for (eigenVectors, dictionary, matrix) in history:
            plt.figure()

            #plot all the articles
            for aId,_ in self.network.articles.items():
                mId = dictionary[aId]
                #pdb.set_trace()
                print dictionary
                print aId
                print mId
                print eigenVectors
                
                plt.scatter([eigenVectors[0,mId]], [eigenVectors[1, mId]], c='r')
                

            #want [-2, 2]
            pch = {-2: "o", -1: "8", 0: "h", 1: "+", 2: "D"}
            for polticalness in range(-2, 3):
                userIds = self.network.getUserIdsWithSpecificPoliticalness(polticalness)
                for uId in userIds:
                    mId = dictionary[uId]
                    plt.scatter(eigenVectors[mId,0], eigenVectors[mId, 1], pch[polticalness])

            for row in range(0, len(matrix)):
                for col in range(0, len(matrix[row])):
                    if matrix[row][col] == 1:
                        plt.plot([eigenVectors[row,0], eigenVectors[col, 0]], [eigenVectors[row,1], eigenVectors[col, 1]], 'k-')

            plt.title("Graph Representation")
            plt.savefig(out_path(self.safe_name + "time=" + str(counter) +".png"))
            plt.close()
            counter = counter + 1

    def save(self, history):
        util.writeCSV(out_path("eigenvectorsgraphrepresentation"), history)