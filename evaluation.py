"""
All metrics should be a subclass of Metric, and at least implement the measure
method.
"""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


import collections
import random

import snap
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import laplacian
import pdb
import util
from util import print_error
from collections import defaultdict

try:
    import networkx as nx
except ImportError:
    print 'install networkx library'
    pass



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

    def measure(self, experiment, network, iterations):
        """Given a Network, return a metric of any type."""
        raise NotImplementedError

    def plot(self, experiment, network, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        print_error("No plotter for %s" % self)

    def save(self, experiment, history):
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


class Statistics(Metric):
    """
    Statistics on graph/articles read:
    Number of articles read by each user
    Number of articles of each type read
    Number of times each article was read (across article types)
    Articles produced by each source
    For each article of the same type, the variety (or nonvariety) of users that read it
    For each specific article, the types of users that read it
    """

    def measure(self, experiment, network, iterations):
        pass

        '''
        userArticleGraph = network.userArticleGraph
        numUsersWithPoliticalness = collections.defaultdict(int)
        distribution = {}
        for user in network.users.itervalues():
            nodeUserId = user.getUserId()
            userPoliticalness = user.getPoliticalness()
            for article in userArticleGraph.GetNI(nodeUserId).GetOutEdges():
                articlePoliticalness = network.getArticle(article).getPoliticalness()
                numUsersWithPoliticalness[userPoliticalness] = numUsersWithPoliticalness[userPoliticalness] + 1
                if userPoliticalness in distribution:
                    innerDict = distribution[userPoliticalness]
                    if articlePoliticalness in innerDict:
                        innerDict[articlePoliticalness] = innerDict[articlePoliticalness] + 1
                    else:
                        innerDict[articlePoliticalness] = 1
                else:
                    distribution[userPoliticalness] = {articlePoliticalness: 1}
        return [distribution, numUsersWithPoliticalness]
        '''


    def plot(self, experiment, network, history):
        # Number of articles read by each user

        userIDs = network.users.keys()
        articleIDs = network.articles.keys()
        numLiked = {userID: 0 for userID in userIDs} # Number of articles each user liked
        timesLiked = {articleID: 0 for articleID in articleIDs} # Number of times each article was liked
        numUserTypes = {2: 0, 1: 0, 0: 0, -1: 0, -2: 0}
        likedFromSource = defaultdict(int)
        POs_of_readers = {articleID: [] for articleID in articleIDs} #articleID: [PO of each user that read]
        # articleIDs = network.articles.keys()
        # userPOs = [network.getUser(userID).politicalness for userID in userIDs]

        # Number of times each article was read by a user of each type (probably should sort articles based on average times read
        # cl, l, n, c, cc dicts: {articleID: times read by users of this type}
        cl = {articleID: 0 for articleID in articleIDs}
        l = {articleID: 0 for articleID in articleIDs}
        n = {articleID: 0 for articleID in articleIDs}
        c = {articleID: 0 for articleID in articleIDs}
        cc = {articleID: 0 for articleID in articleIDs} # -2

        timesReadByType = {2: cl, 1: l, 0: n, -1: c, -2: cc}

        for userID in userIDs:
            numLiked[userID] = 0
            userPO = network.getUser(userID).politicalness
            numUserTypes[userPO] += 1
            for article in network.articlesLikedByUser(userID):
                numLiked[userID] += 1
                timesLiked[article.articleId] += 1
                timesReadByType[userPO][article.articleId] += 1
                likedFromSource[article.getSource()] += 1
                POs_of_readers[article.articleId].append(userPO)

        print "userID: number of articles liked"
        # print numLiked

        print "articleID: number of users liked"
        # print timesLiked

        print "articleID: POs of users that liked"
        print POs_of_readers

        print "source: number of times an article was liked from this source"
        print likedFromSource



        print self.name
        plt.figure()
        plt.plot(range(0, len(numLiked.keys())), sorted(numLiked.values()))
        plt.xlabel("Ordered Users")
        plt.ylabel("Number of Articles Liked")
        plt.title("Number of Articles Liked By Each User \n " + str(experiment.parameters))
        plt.savefig(experiment.out_path(self.safe_name + " NumArticlesLiked" + ".png"))
        plt.close()

        '''
        print self.name
        plt.figure()
        plt.plot(range(0, len(likedFromSource.keys())), likedFromSource.keys())
        plt.xticks(range(0, len(likedFromSource.keys())), likedFromSource.keys())
        plt.xticks(range(0, len(likedFromSource.keys())), likedFromSource.keys(), rotation=45)  # writes strings with 45 degree angle
        plt.xlabel("Source")
        plt.ylabel("Number of Articles Liked from Source")
        plt.title("Number of Articles Liked from Each Source \n " + str(experiment.parameters))
        plt.savefig(experiment.out_path(self.safe_name + " LikedFromSource" + ".png"))
        plt.close()
        '''

        plt.figure()
        plt.plot(range(0, len(timesLiked.keys())), sorted(timesLiked.values()))
        plt.xlabel("Ordered Articles")
        plt.ylabel("Number of Times Article Liked")
        plt.title("Number of Users that Like Each Article \n " + str(experiment.parameters))
        plt.savefig(experiment.out_path(self.safe_name + " TimesArticlesLiked" + ".png"))
        plt.close()

        # Number of times each article was read by a user of each type (probably should sort articles based on average times read
        # cl, l, n, c, cc dicts: {articleID: times read by users of this type}
        plt.figure()
        plt.plot(cl.keys(), cl.values(), 'bx', l.keys(), l.values(), 'gx', n.keys(), n.values(), 'kx',
                 c.keys(), c.values(), 'cx', cc.keys(), cc.values(), 'rx')
        plt.legend(["consistently liberal", "mostly liberal", "mixed", "mostly conservative", "consistently conservative"])
        plt.xlabel("Article")
        plt.ylabel("Number of Times Article Liked")
        plt.title("Number of Users of Each Type that Like Each Article \n " + str(experiment.parameters))
        plt.savefig(experiment.out_path(self.safe_name + " NumTypesThatReadArticle" + ".png"))
        plt.close()

        # ["consistently liberal", "mostly liberal", "mixed", "mostly conservative", "consistently conservative"]
        # Number of users of each type
        plt.figure()
        plt.plot(numUserTypes.keys(), numUserTypes.values(), 'kx')
        # plt.legend(
        #    ["consistently liberal", "mostly liberal", "mixed", "mostly conservative", "consistently conservative"])
        plt.xlabel("Type")
        plt.ylabel("Number of Users of Type")
        plt.title("Number of Users of Each Type \n " + str(experiment.parameters))
        plt.savefig(experiment.out_path(self.safe_name + " NumTypes" + ".png"))
        plt.close()



        # Variance of POs of users that liked each article, ordered by variance of article
        # dict of variances
        reader_PO_variance = {articleID: 0 for articleID in articleIDs}
        for articleID in reader_PO_variance.keys():
            reader_PO_variance[articleID] = np.var(POs_of_readers[articleID])

        IDs, vars = zip(*sorted(zip(reader_PO_variance.keys(), reader_PO_variance.values()), key = lambda x: x[1]))

        numLikes = []
        # Build list of number of likes ordered by same ordering as var
        for i in range(0, len(IDs)):
            numLikes.append(timesLiked[IDs[i]])

        print reader_PO_variance
        print "IDs"
        print IDs
        print "vars"
        print vars

        # Plot variance of POs of those who have liked each article, ordered by var
        '''
        plt.figure()
        plt.plot(range(0, len(vars)), 0.0 + (np.array(numLikes)/len(userIDs)), 'kx', range(0, len(vars)), vars, 'r-')
        # plt.legend(
        #    ["consistently liberal", "mostly liberal", "mixed", "mostly conservative", "consistently conservative"])
        plt.xlabel("Sorted Article ID")
        plt.ylabel("Variance in Users")
        plt.title("Variance in Pol. Orientations of Likers of Each Article \n " + str(experiment.parameters))
        plt.legend(["Perc. of Users Who Liked Article", "Variance in Pol. Orient. of Users"])

        plt.savefig(experiment.out_path(self.safe_name + " LikerVar" + ".png"))
        plt.close()
        '''


        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(range(0, len(vars)), numLikes, 'kx')
        ax2.plot(range(0, len(vars)), vars, 'r-')

        ax1.set_xlabel('Ordered ArticleID')
        ax1.set_ylabel('Number of Users Who Liked Article', color='k')
        ax2.set_ylabel('Variance in Pol. Orient. of Users Who Liked Article', color='r')
        plt.title("Variance in Pol. Orientations of Likers of Each Article \n " + str(experiment.parameters))
        plt.savefig(experiment.out_path(self.safe_name + " LikerVar" + ".png"))
        plt.close()

            # Number of times pair of users read same article
        # Number of times user read an article from each source

        # Number of users that read each article
        # Types of users that read each article (skewedness in distribution in types of user that read each article?)
        # Looking at the top two articles, orientations of users that read those two articles
        # Variance in the readership; how much does distribution of pol.orient of users vary across each article
        # Variance in who read each article



    def save(self, experiment, history):
        pass
        util.writeCSV(experiment.out_path("statistics"), history)

class CliquePercolation(Metric):

    def measure(self, experiment, network, iterations):
        return []

    def plot(self, experiment, network, history):
        print "start creating user user graph"
        G, _ , _ = network.createUserUserGraph()
        print "finished creating user user graph"
        
        cliques = nx.k_clique_communities(G, 5)
        values = []
        for c in cliques:
            dictionary = collections.defaultdict(int)
            for NI in c:
                politicalness = 0
                if NI in network.users:
                    politicalness = network.users[NI].getPoliticalness()
                dictionary[politicalness] = dictionary[politicalness] + 1
            values.append(dictionary)
        for i, h in enumerate(values):
            val = []
            for pol in range(-2, 3):
                val.append(h[pol])
            try:
                print self.name
                plt.figure()
                plt.bar(range(-2, 3), val)
                plt.xlabel("Politicalness")
                plt.ylabel("Count")
                plt.title("Count vs. Politicalness Community = " + str(i))
                plt.savefig(util.out_path(self.safe_name  + "community=" + str(i) + '.png', "CliquePercolation"))
                plt.close()
            except IOError:
                print_error("Error making plot")




class WeightedGirvanNewman(Metric):

    def measure(self, experiment, network, iterations):
        #https://networkx.github.io/documentation/development/reference/generated/networkx.algorithms.centrality.betweenness_centrality.html
        # if iterations >= experiment.numIterations -1:
        #     print "start creating user user graph"
        #     G, _ , _ = network.createUserUserGraph()
        #     print "finished creating user user graph"
        #     betweenessCentr = nx.edge_betweenness_centrality(G, normalized=True, weight="weight")
        #     numEdgesRemoved = 0
        #     betweeness = []
        #     for key, value in betweenessCentr.items():
        #         betweeness.append((key, value))
        #     sorted(betweeness, key = lambda x: x[1], reverse = True)
        #     while nx.number_connected_components(G) != 5 and numEdgesRemoved < min(30, len(betweeness)):
        #         G.remove_edge(betweeness[numEdgesRemoved][0][0], betweeness[numEdgesRemoved][0][1])
        #         numEdgesRemoved = numEdgesRemoved + 1
        #     components = nx.connected_component_subgraphs(G)
        #     values = []
        #     for CnCom in components:
        #         dictionary = collections.defaultdict(int)
        #         for NI in CnCom:
        #             politicalness = 0
        #             if NI in network.users:
        #                 politicalness = network.users[NI].getPoliticalness()
        #             elif NI in network.articles:
        #                 politicalness = network.articles[NI].getPoliticalness()
        #             else:
        #                 raise Exception("Should not reach here")
        #             dictionary[politicalness] = dictionary[politicalness] + 1
        #         values.append(dictionary)
        #     return [betweenessCentr, values]
        # else:
        return [None, None]

    def plot(self, experiment, network, history):
        print "start creating user user graph"
        G, _ , _ = network.createUserUserGraph()
        print "finished creating user user graph"
        betweenessCentr = nx.edge_betweenness_centrality(G, normalized=True, weight="weight")
        numEdgesRemoved = 0
        betweeness = []
        for key, value in betweenessCentr.items():
            betweeness.append((key, value))
        sorted(betweeness, key = lambda x: x[1], reverse = True)
        while nx.number_connected_components(G) != 5 and numEdgesRemoved < min(30, len(betweeness)):
            G.remove_edge(betweeness[numEdgesRemoved][0][0], betweeness[numEdgesRemoved][0][1])
            numEdgesRemoved = numEdgesRemoved + 1
        components = nx.connected_component_subgraphs(G)
        values = []
        for CnCom in components:
            dictionary = collections.defaultdict(int)
            for NI in CnCom:
                politicalness = 0
                if NI in network.users:
                    politicalness = network.users[NI].getPoliticalness()
                elif NI in network.articles:
                    politicalness = network.articles[NI].getPoliticalness()
                else:
                    raise Exception("Should not reach here")
                dictionary[politicalness] = dictionary[politicalness] + 1
            values.append(dictionary)
        print self.name
        plt.figure()
        plt.plot(sorted(betweenessCentr.values()))
        plt.xlabel("Edge Ordering")
        plt.ylabel("Weighted Edge Betweeness")
        plt.title("Weighted Edge Betweeness")
        plt.savefig(experiment.out_path(self.name + ".png"))
        plt.close()
        #politicalnessByCommunityHistory = history[-1][1]
        #politicalnessByCommunityHistory = politicalnessByCommunityHistory[(len(politicalnessByCommunityHistory)-5):len(politicalnessByCommunityHistory)]
        
        for i, h in enumerate(values):
            val = []
            for pol in range(-2, 3):
                val.append(h[pol])
            try:
                print self.name
                plt.figure()
                plt.bar(range(-2, 3), val)
                plt.xlabel("Politicalness")
                plt.ylabel("Count")
                plt.title("Count vs. Politicalness Community = " + str(i))
                plt.savefig(util.out_path(self.safe_name  + "community=" + str(i) + '.png', "Modularity2"))
                plt.close()
            except IOError:
                print_error("Error making plot")

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path(self.safe_name), history)



class GraphViz(Metric):
    """
    Visualize graph
    """

    def measure(self, experiment, network, iterations):
        print self.name

        plt.figure()
        G = nx.Graph()  # Create a graph
        userIDs = network.users.keys()
        articleIDs = network.articles.keys()
        userPOs = [network.getUser(userID).politicalness for userID in userIDs]


        for userID in userIDs:
            G.add_node(userID) #network.getUser(userID).politicalness)
            #G.node[userID]['color'] = 'blue'


        for articleID in articleIDs:
            G.add_node(articleID)

        # print userIDs
        # print userPOs

        '''
        # nodes
nx.draw_networkx_nodes(G,pos,
                       nodelist=[0,1,2,3],
                       node_color='r',
                       node_size=500,
                   alpha=0.8)
nx.draw_networkx_nodes(G,pos,
                       nodelist=[4,5,6,7],
                       node_color='b',
                       node_size=500,
                   alpha=0.8)

        # Draw user nodes and color on polit.orient.

        nx.draw_networkx_nodes(G,pos,
                       nodelist=[0,1,2,3],
                       node_color='r',
                       node_size=500,
                   alpha=0.8)

# edges
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edges(G,pos,
                       edgelist=[(0,1),(1,2),(2,3),(3,0)],
                       width=8,alpha=0.5,edge_color='r')
nx.draw_networkx_edges(G,pos,
                       edgelist=[(4,5),(5,6),(6,7),(7,4)],
                       width=8,alpha=0.5,edge_color='b')
        '''


        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist= userIDs, node_color= userPOs) # pos,
                               # nodelist= userIDs, #[0, 1, 2, 3],
                               # node_color= userPOs) # 'r',
                               # node_size=500,
                               # alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=articleIDs, node_color='red')

        for userID in userIDs:
            for article in network.articlesLikedByUser(userID):
                G.add_edge(userID, article.articleId)  # Add an edge for each dictionary entry
            # Nodes are automatically created

        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)


        # pos = nx.shell_layout(G)  # Layout algorithm
        # nx.draw_circular(G, with_labels=True)  # Draw the graph

        plt.title("Graph at Iteration " + str(iterations))
        plt.savefig(experiment.out_path(self.safe_name + "Iter" + str(iterations) + ".png"))
        plt.close()
        pass


    def plot(self, experiment, network, history):
        pass


    def save(self, experiment, history):
        pass
        util.writeCSV(experiment.out_path("statistics"), history)


# No longer makes sense since articles don't have political leanings anymore.
class ReadingDistribution(Metric):
    """
    Distribution of article political leaning given user political leaning.
    """
    def measure(self, experiment, network, iterations):
        userArticleGraph = network.userArticleGraph
        numUsersWithPoliticalness = collections.defaultdict(int)
        distribution = {}
        for user in network.users.itervalues():
            nodeUserId = user.getUserId()
            userPoliticalness = user.getPoliticalness()
            for article in userArticleGraph.GetNI(nodeUserId).GetOutEdges():
                articlePoliticalness = network.getArticle(article).getPoliticalness()
                numUsersWithPoliticalness[userPoliticalness] = numUsersWithPoliticalness[userPoliticalness] + 1
                if userPoliticalness in distribution:
                    innerDict = distribution[userPoliticalness]
                    if articlePoliticalness in innerDict:
                        innerDict[articlePoliticalness] = innerDict[articlePoliticalness] + 1
                    else:
                        innerDict[articlePoliticalness] = 1
                else:
                    distribution[userPoliticalness] = {articlePoliticalness: 1}
        return [distribution, numUsersWithPoliticalness]

    def plot(self, experiment, network, history):
        last = history[-1][0]
        for key, value in last.items():
            #value is a dictionary
            keys = []
            vals = []
            for k1, v1 in value.items():
                keys.append(k1)
                vals.append(v1)
            print self.name
            plt.figure()
            plt.bar(keys, vals, color = "blue")
            plt.xlabel("Article Politicalness")
            plt.ylabel("Frequency")
            plt.title("Which Articles do Users with politicalness " + str(key) + " Read")
            #make this a mosaic plot later
            plt.savefig(experiment.out_path(self.safe_name + "key=" + str(key) + ".png"))
            plt.close()
        numUsersWithPoliticalness = history[-1][1]
        for key, value in last.items():
            #value is a dictionary
            keys = []
            vals = []
            for k1, v1 in value.items():
                keys.append(k1)
                if numUsersWithPoliticalness[k1] != 0:
                    vals.append(v1 / (1.0 * numUsersWithPoliticalness[k1]))
                else:
                    vals.append(0)
            print self.name
            plt.figure()
            plt.bar(keys, vals, color = "blue")
            plt.xlabel("Article Politicalness")
            plt.ylabel("Frequency Normalized bby number of users")
            plt.title("Which Articles do Users with politicalness " + str(key) + " Read")
            #make this a mosaic plot later
            plt.savefig(experiment.out_path(self.safe_name + "Normalized key=" + str(key) + ".png"))
            plt.close()

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("readingDistribution"), history)


class PathsBetweenPoliticalnesses(Metric):
    def __init__(self, politicalness1=-2, politicalness2=2):
        self.politicalness1 = politicalness1
        self.politicalness2 = politicalness2

    def measure(self, experiment, network, iterations):
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

    def plot(self, experiment, network, history):
        print self.name
        plt.figure()
        plt.plot(history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Average Distance Between Politicalness")
        plt.title("Average Distance Between " + str(self.politicalness1) + " and " + str(self.politicalness2))
        plt.savefig(util.out_path(self.safe_name + '.png', "PathsBetweenPoliticalnesses"))

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("pathsbetweenpoliticalness" + str(self.politicalness1) + str(self.politicalness2)), history)


class Modularity2(Metric):
    def measure(self, experiment, network, iterations):
        CmtyV = snap.TCnComV()
        modularity = snap.CommunityGirvanNewman(network.userArticleGraph, CmtyV)
        politicalnessByCommunity = {}
        for Cmty in CmtyV:
            for NI in Cmty:
                politicalness = 0
                if NI in network.users:
                    politicalness = network.users[NI].getPoliticalness()
                elif NI in network.articles:
                    politicalness = network.articles[NI].getPoliticalness()
                else:
                    raise Exception("Error in finding politicalness")
                if Cmty in politicalnessByCommunity:
                    innerDict = politicalnessByCommunity[Cmty]
                    if politicalness in innerDict:
                        politicalnessByCommunity[Cmty][politicalness] = politicalnessByCommunity[Cmty][politicalness] + 1
                    else:
                        politicalnessByCommunity[Cmty][politicalness] = 1
                else:
                    politicalnessByCommunity[Cmty] = collections.defaultdict(int)
                    politicalnessByCommunity[Cmty][politicalness] = 1

        return [modularity, politicalnessByCommunity]

    def plot(self, experiment, network, history):
        print self.name
        modularity = map(lambda x: x[0], history)
        plt.figure()
        plt.plot(modularity)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Modularity of whole netowrk")
        plt.savefig(experiment.out_path(self.safe_name + ".png"))
        politicalnessByCommunityHistory = map(lambda x: x[1], history)
        politicalnessByCommunityHistory = politicalnessByCommunityHistory[(len(politicalnessByCommunityHistory)-5):len(politicalnessByCommunityHistory)]
        for i, h in enumerate(politicalnessByCommunityHistory):
            for cmty, innerDict in h.items():
                values = []
                for pol in range(-2, 3):
                    values.append(innerDict[pol])
                try:
                    print self.name
                    plt.figure()
                    plt.bar(range(-2, 3), values)
                    plt.xlabel("Politicalness")
                    plt.ylabel("Count")
                    plt.title("Count vs. Politicalness Community = " + str(cmty))
                    plt.savefig(experiment.out_path(self.safe_name + "community = " + str(cmty) + "iterations=" + str(i+len(modularity) -5) + '.png', "Modularity2"))
                    plt.close()
                except IOError:
                    print_error("Error making plot")

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("modularity2"), history)


class Modularity(Metric):
    def measure(self, experiment, network, iterations):
        
        def modularity(graph):
            result = []
            for idx, i in enumerate(range(-2, 3)):
                ids = network.getUserIdsWithSpecificPoliticalness(i)
                Nodes = snap.TIntV()
                for ni in ids:
                    Nodes.Add(ni)
                result.append(snap.GetModularity(graph, Nodes))

            return result
        return [modularity(network.userArticleGraph), modularity(network.userArticleFriendGraph), modularity(network.createUserUserGraph()[1])]

    def plot(self, experiment, network, history):
        
        def plotHelper(history, id):
            for idx, i in enumerate(range(-2, 3)):
                print self.name
                plt.figure()
                oneCluster = map(lambda x:x[idx], history)
                plt.plot(oneCluster)
                plt.savefig(util.out_path(self.safe_name + 'politicalness' + str(i) + id + '.png', "Modularity"))
                plt.close()

        ids = ["userArticleGraph", "userArticleFriendGraph", "userUserGraph"]
        for i, id in enumerate(ids):
            plotHelper(map(lambda x: x[i], history), id)

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("modularity"), history)


# class ModularityWRTFriends(Metric):
#     def measure(self, experiment, network, iterations):
#         result = []
#         for idx, i in enumerate(range(-2, 3)):
#             ids = network.getUserIdsWithSpecificPoliticalness(i)
#             Nodes = snap.TIntV()
#             for ni in ids:
#                 Nodes.Add(ni)
#             result.append(snap.GetModularity(network.userArticleFriendGraph, Nodes))

#         return result

#     def plot(self, experiment, network, history):
#         for idx, i in enumerate(range(-2, 3)):
#             print self.name
#             plt.figure()
#             oneCluster = map(lambda x:x[idx], history)
#             plt.plot(oneCluster)
#             plt.savefig(experiment.out_path(self.safe_name + 'politicalness' + str(i) + '.png', "Modularity"))
#             plt.close()

#     def save(self, experiment, history):
#         util.writeCSV(experiment.out_path("modularity"), history)


def copyGraph(graph):
    copyGraph = snap.TUNGraph.New()
    for node in graph.Nodes():
        copyGraph.AddNode(node.GetId())
    for edge in graph.Edges():
        copyGraph.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
    return copyGraph


class Betweenness(Metric):
    def measure(self, experiment, network, iterations):
        
        def betweeeness(graph):
            Nodes = snap.TIntFltH()
            Edges = snap.TIntPrFltH()
            snap.GetBetweennessCentr(graph, Nodes, Edges, 1.0)
            # for edge in Edges:
            #   		print "edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge])

            betweenessCentr = []
            for edge in Edges:
                betweenessCentr.append((edge, Edges[edge]))
            betweenessCentr.sort(key = lambda x: x[1], reverse = True)
            copyOfGraph = copyGraph(graph)
            components = snap.TCnComV()
            snap.GetWccs(copyOfGraph, components)
            numEdgesRemoved = 0

            while len(components) != 5 and numEdgesRemoved < min(30, len(betweenessCentr)):
                copyOfGraph.DelEdge(betweenessCentr[numEdgesRemoved][0].GetVal1(), betweenessCentr[numEdgesRemoved][0].GetVal2())
                components = snap.TCnComV()
                snap.GetWccs(copyOfGraph, components)
                numEdgesRemoved = numEdgesRemoved + 1
            components = snap.TCnComV()
            snap.GetWccs(copyOfGraph, components)
            values = []
            for CnCom in components:
                dictionary = collections.defaultdict(int)
                for NI in CnCom:
                    politicalness = 0
                    if NI in network.users:
                        politicalness = network.users[NI].getPoliticalness()
                    elif NI in network.articles:
                        #politicalness = network.articles[NI].getPoliticalness()
                        print ""
                    else:
                        raise Exception("Should not reach here")
                    dictionary[politicalness] = dictionary[politicalness] + 1
                values.append(dictionary)
            return [betweenessCentr, values]

        return [betweeeness(network.userArticleGraph), betweeeness(network.userArticleFriendGraph), betweeeness(network.createUserUserGraph()[1])]

    def plot(self, experiment, network, history):

        def plotHelper(history, id):
            betweeness = map(lambda x: x[0], history)
            betweeness = betweeness[(len(betweeness)-10):len(betweeness)]
            for i,b in enumerate(betweeness):
                plt.figure()
                print self.name
                plt.plot(sorted(map(lambda x: x[1], b)))
                plt.xlabel("Edge Ordering")
                plt.ylabel("Edge Betweenness")
                plt.title("Edge Betweeness")
                plt.savefig(experiment.out_path(self.safe_name + "iteartions=" + str(i) + '.png'))
                plt.close()
            values = map(lambda x: x[1], history)
            values = values[(len(values)-5):len(values)]
            for i, innerDict in enumerate(values):
                for j,v in enumerate(innerDict):
                    if j < 15:
                        val = []
                        for pol in range(-2, 3):
                            val.append(v[pol])
                        plt.figure()
                        plt.bar(range(-2, 3), val)
                        plt.xlabel("Politicalness")
                        plt.ylabel("Count")
                        plt.title("Count vs. Politicalness Community = "+ str(i))
                        plt.savefig(util.out_path(self.safe_name + "Community " + str(j) + " iterations=" + str(i) + id + '.png', "Betweenness_Community"))
                        plt.close()

        ids = ["userArticleGraph", "userArticleFriendGraph", "userUserGraph"]
        for i, id in enumerate(ids):
            plotHelper(map(lambda x: x[i], history), id)

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("modularity2"), history)

# class BetweennessWRTFriends(Betweenness):
#     def measure(self, experiment, network, iterations):
#         Nodes = snap.TIntFltH()
#         Edges = snap.TIntPrFltH()
#         snap.GetBetweennessCentr(network.userArticleFriendGraph, Nodes, Edges, 1.0)

#         betweenessCentr = []
#         for edge in Edges:
#             betweenessCentr.append((edge, Edges[edge]))
#         betweenessCentr.sort(key = lambda x: x[1], reverse = True)
#         copyOfGraph = copyGraph(network.userArticleGraph)
#         components = snap.TCnComV()
#         snap.GetWccs(copyOfGraph, components)
#         numEdgesRemoved = 0

#         while len(components) != 5 and numEdgesRemoved < min(30, len(betweenessCentr)):
#             copyOfGraph.DelEdge(betweenessCentr[numEdgesRemoved][0].GetVal1(), betweenessCentr[numEdgesRemoved][0].GetVal2())
#             components = snap.TCnComV()
#             snap.GetWccs(copyOfGraph, components)
#             numEdgesRemoved = numEdgesRemoved + 1
#         components = snap.TCnComV()
#         snap.GetWccs(copyOfGraph, components)
#         values = []
#         for CnCom in components:
#             dictionary = collections.defaultdict(int)
#             for NI in CnCom:
#                 politicalness = 0
#                 if NI in network.users:
#                     politicalness = network.users[NI].getPoliticalness()
#                 elif NI in network.articles:
#                     politicalness = network.articles[NI].getPoliticalness()
#                 else:
#                     raise Exception("Should not reach here")
#                 dictionary[politicalness] = dictionary[politicalness] + 1
#             values.append(dictionary)
#         return [betweenessCentr, values]


class UserDegreeDistribution(Metric):
    def __init__(self, politicalness="all"):
        self.politicalness = politicalness

    def measure(self, experiment, network, iterations):
        userArticleGraph = network.userArticleGraph
        degree = []
        for user in network.users.itervalues():
            uId = user.getUserId()
            if self.politicalness == "all" or (str(user.getPoliticalness()) == self.politicalness):
                degree.append(userArticleGraph.GetNI(uId).GetOutDeg())
        return degree

    def plot(self, experiment, network, history):
        # Just plot the dist for the last iteration
        print self.name
        last = history[-1]
        plt.figure()
        plt.hist(last)
        plt.xlabel("User Degree")
        plt.ylabel("Frequency")
        plt.title("Histogram of User Degree")
        plt.savefig(experiment.out_path(self.safe_name + '.png'))
        plt.close()

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("userDegree"), history)


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

    def measure(self, experiment, network, iterations):
        return map(lambda x: x[1], getArticleDegreeDistribution(network, self.article_type))

    def plot(self, experiment, network, history):
        for i,h in enumerate(history):
            plt.figure()
            plt.hist(h)
            plt.xlabel("Aricle Degree")
            plt.ylabel("Frequency")
            plt.title("Histogram of Article Degree")
            plt.savefig(util.out_path(self.safe_name + self.article_type + "time=" + str(i) + '.png', "AritcleDegree"))
            plt.close()

    def save(self, experiment, history):
        # This is for Akhil's R plots
        if self.article_type == 'all':
            util.writeCSV(experiment.out_path("articleDegree"), history)
        elif self.article_type == 'alive':
            util.writeCSV(experiment.out_path("aliveAricleDegree"), history)
        elif self.article_type == 'dead':
            util.writeCSV(experiment.out_path("deadArticle"), history)


class DistributionOfLifeTime(Metric):

    def __init__(self, article_type):
        self.article_type = article_type

    def measure(self, experiment, network, iterations):
        lifeTime = []
        for article in network.articles.itervalues():
            if self.article_type == "alive" and not article.getIsDead():
                lifeTime.append(article.getTimeToLive() - iterations)
            if self.article_type == "dead" and article.getIsDead():
                lifeTime.append(article.getTimeToLive() - iterations)
        return lifeTime

    def plot(self, experiment, network, history):
        for i,h in enumerate(history):
            print self.name
            plt.figure()
            plt.hist(h)
            plt.xlabel("Aricle Lifetime")
            plt.ylabel("Frequency")
            plt.title("Histogram of Article Lifetime")
            plt.savefig(util.out_path(self.safe_name + self.article_type + "time=" + str(i) + '.png', "LifetimeDistribution"))
            plt.close()

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("article lifetime disitburtion" + self.article_type), history)



class AliveArticles(Metric):
    def measure(self, experiment, network, iterations):
        counterAlive = 0
        counterDead = 0
        for article in network.articles.itervalues():
            if not article.getIsDead():
                counterAlive = counterAlive + 1
            else:
                counterDead = counterDead + 1
        return counterAlive

    def plot(self, experiment, network, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        numIterations = len(history)
        print self.name
        plt.figure()
        plt.plot(range(0, numIterations), history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Number of Alive Articles")
        plt.title("Number of Alive Articles vs. Number of Iterations")
        plt.savefig(experiment.out_path(self.safe_name + '.png'))
        plt.close()

    def save(self, experiment, history):
        """
        Save history to a file.
        """
        util.writeCSV(experiment.out_path("numberAliveArticles"), history)

class OverallClustering(Metric):
    def measure(self, experiment, network, iterations):
        #printGraph(network.userArticleGraph)
        if iterations % 50 == 0:
            return [snap.GetClustCf(network.userArticleGraph, -1), snap.GetClustCf(network.userArticleFriendGraph, -1), snap.GetClustCf(network.createUserUserGraph()[1], -1)]
        return [snap.GetClustCf(network.userArticleGraph, -1), snap.GetClustCf(network.userArticleFriendGraph, -1), -1]

    def plot(self, experiment, network, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        print self.name
        ids = ["userArticleGraph", "userArticleFriendGraph", "userUserGraph"]

        def plotHelper(history, id):
            plt.figure()
            plt.plot(history)
            plt.xlabel("Number of Iterations")
            plt.ylabel("Clustering Coefficient")
            plt.title("Clustering Coefficient vs. Number of Iterations")
            plt.savefig(experiment.out_path(self.safe_name + id + '.png'))
            plt.close()

        for i, id in enumerate(ids):
            plotHelper(map(lambda x: x[i], history), id)

    def save(self, experiment, history):
        """
        Save history to a file.
        """
        util.writeCSV(experiment.out_path("OverallClustering" + self.name), history)

# class OverallClusteringWRTFriends(OverallClustering):
#     def measure(self, experiment, network, iterations):
#         #printGraph(network.userArticleGraph)
#         return snap.GetClustCf(network.userArticleFriendGraph, -1)    


class DeadArticles(Metric):
    def measure(self, experiment, network, iterations):
        counterDead = 0
        for article in network.articles.itervalues():
            if article.getIsDead():
                counterDead = counterDead + 1
        return counterDead

    def plot(self, experiment, network, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        numIterations = len(history)
        print self.name
        plt.figure()
        plt.plot(range(0, numIterations), history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Number of Dead Articles")
        plt.title("Number of Dead Articles vs. Number of Iterations")
        plt.savefig(experiment.out_path(self.safe_name + '.png'))
        plt.close()

    def save(self, experiment, history):
        """
        Save history to a file.
        """
        util.writeCSV(experiment.out_path("numberDeadArticles"), history)

def printGraph(graph):
    for EI in graph.Edges():
        print "edge (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())

class ClusterPoliticalness(Metric):

    def __init__(self, politicalness):
        self.politicalness = politicalness

    # triangles
    def clusterOneNode(self, node, graph):
        #printGraph(graph)
        degree = node.GetOutDeg()
        if degree < 2:
            return 0
        counter = 0
        for i in range(node.GetDeg()):
            id = node.GetNbrNId(i) 
            for k in node.GetOutEdges():
                if graph.IsEdge(k, id):
                    counter = counter + 1
        counter = counter / 2
        return (2.0 * counter) / (degree * (degree - 1))        

    def measure(self, experiment, network, iterations):
        
        def clusterPolticalness(graph):
            cluster = []
            for user in network.users.itervalues():
                #if iterations > 35 and self.politicalness == "all":
                    #pdb.set_trace()
                if self.politicalness == "all" or str(
                        user.getPoliticalness()) == self.politicalness:
                    result = self.clusterOneNode(
                        graph.GetNI(user.getUserId()), graph)
                    cluster.append(result)
            return mean(cluster)
        if iterations % 40 == 0:
            return [clusterPolticalness(network.userArticleGraph), clusterPolticalness(network.userArticleFriendGraph), clusterPolticalness(network.createUserUserGraph()[1])]
        return [clusterPolticalness(network.userArticleGraph), clusterPolticalness(network.userArticleFriendGraph), -1]

    def plot(self, experiment, network, history):
        """
        Given a list of objects of the type returned by self.measure, make an
        appropriate plot of this metric over time.
        """
        def plotHelper(history, id):
            numIterations = len(history)
            print self.name
            plt.figure()
            plt.plot(range(0, numIterations), history)
            plt.xlabel("Number of Iterations")
            plt.ylabel("Clustering Coefficient")
            plt.title("Clustering Coefficient for politicalness " + str(self.politicalness) + "\n vs. Number of Itertions")
            plt.savefig(experiment.out_path(self.safe_name + "polticialness" + str(self.politicalness) + id + '.png'))
            plt.close()

        ids = ["userArticleGraph", "userArticleFriendGraph", "userUserGraph"]
        for i, id in enumerate(ids):
            plotHelper(map(lambda x: x[i], history), id)

    def save(self, experiment, history):
        """
        Save history to a file.
        """
        util.writeCSV(experiment.out_path("clusterPoliticalness" + "_politicalness=" + self.politicalness + self.safe_name), history)

# class ClusterPoliticalnessWRTFriends(ClusterPoliticalness):

#     def measure(self, experiment, network, iterations):
#         userArticleGraph = network.userArticleGraph
#         cluster = []
#         for user in network.users.itervalues():
#             #if iterations > 35 and self.politicalness == "all":
#                 #pdb.set_trace()
#             if self.politicalness == "all" or str(
#                     user.getPoliticalness()) == self.politicalness:
#                 result = self.clusterOneNode(
#                     userArticleGraph.GetNI(user.getUserId()), network.userArticleFriendGraph)
#                 cluster.append(result)
#         return mean(cluster)


class LargestConnectedComponent(Metric):

    def measure(self, experiment, network, iterations):
        Components = snap.TCnComV()
        snap.GetWccs(network.userArticleGraph, Components)
        numComponents = []
        for CnCom in Components:
            numComponents.append(CnCom.Len())
        return numComponents

    def plot(self, experiment, network, history):
        for i,elem in enumerate(history):
            print self.name
            plt.figure()
            plt.bar(range(len(elem)), elem)
            plt.savefig(util.out_path(self.safe_name + "connected_compoenents_" + str(i) + '.png', "LargestConnectedComponent"))
            plt.close()
        largestComponent = map(max, history)
        plt.figure()
        plt.plot(largestComponent)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Size of Largest Connected Component")
        plt.title("Size of Largest Connected Component vs. Number of Iterations")
        plt.savefig(experiment.out_path(self.safe_name + "largest_component" + ".png"))
        plt.clf()
        plt.close()
        numComponents = map(len, history)
        plt.figure()
        plt.plot(numComponents)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Number of Components")
        plt.title("Number of Components vs. Number of Iterations")
        plt.savefig(experiment.out_path(self.safe_name + "number_components" + ".png"))
        plt.clf()
        plt.close()

    def save(self, experiment, history):
        largestComponent = map(max, history)
        util.writeCSV(experiment.out_path("largestCompeonent"), history)
        numComponents = map(len, history)
        util.writeCSV(experiment.out_path("numberCompeonent"), history)


class EigenVectors(Metric):

    def measure(self, experiment, network, iterations):
        
        def eigenvector(graph):
            try:
                EigvV =  snap.TFltV()
                snap.GetEigVec(graph, EigvV)
                result = []
                for Val in EigvV:
                    result.append(Val)
                return sorted(result)
            except:
                return []
        return [eigenvector(network.userArticleGraph), eigenvector(network.userArticleFriendGraph), eigenvector(network.createUserUserGraph()[1])]

    def plot(self, experiment, network, history):
        
        def plotHelper(history, id):
            last = history[-1]
            print self.name
            plt.figure()
            plt.plot(last)
            plt.xlabel("Rank of Eigenvector")
            plt.ylabel("Values of Eigenvector")
            plt.title("First Eigenvector")
            plt.savefig(experiment.out_path(self.safe_name + id + ".png"))
            plt.close()

        ids = ["userArticleGraph", "userArticleFriendGraph", "userUserGraph"]
        for i, id in enumerate(ids):
            plotHelper(map(lambda x: x[i], history), id)

def getEigenVectorEigenValue(network, graph, iterations):
    matrix, uIdOrAIdToMatrix = network.calcAdjacencyMatrix(graph)

    matrixIdPoliticalness = []
    for uId, user in network.users.items():
        matrixId = uIdOrAIdToMatrix[uId] 
        matrixIdPoliticalness.append([matrixId, user.getPoliticalness()])
    for aId, article in network.articles.items():
        matrixId = uIdOrAIdToMatrix[aId]
        matrixIdPoliticalness.append([matrixId, article.getPoliticalness()])
    util.writeCSV(experiment.out_path("matrixId_topolitcaless iterations=" + str(iterations)), matrixIdPoliticalness)
    #print matrix
    #print len(matrix)
    #print len(matrix[0])
    laplacian = scipy.sparse.csgraph.laplacian(np.array(matrix))
    eigenvalue, eigenvector = np.linalg.eig(laplacian)
    #print eigenvalue
    #print eigenvector
    #result = [x for (y,x) in sorted(zip(eigenvalue,eigenvector))]
    eigenvalueIdx = eigenvalue.argsort()
    result = eigenvector[:, eigenvalueIdx]
    return (result, uIdOrAIdToMatrix, matrix)

# class EigenVectorsWRTFriends(EigenVectors):

#     def measure(self, experiment, network, iterations):
#         EigvV =  snap.TFltV()
#         snap.GetEigVec(network.userArticleFriendGraph, EigvV)
#         result = []
#         for Val in EigvV:
#             result.append(Val)
#         return sorted(result)


class MoreEigenVectors(Metric):

    def measure(self, experiment, network, iterations):
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
        # eigenvalue, eigenvector = np.linalg.eig(laplacian)
        # result = [x for (y,x) in sorted(zip(eigenvalue,eigenvector))]
        result = getEigenVectorEigenValue(network, network.userArticleGraph, iterations)
        util.writeCSV(experiment.out_path("adjacencyMatrix_iterations=" + str(iterations)), result[2])
        eigenvector = result[0]
        return eigenvector[:,1]

    def plot(self, experiment, network, history):
        for i,eigenvector in enumerate(history):
            if not eigenvector is None:
                sortedEigenvector = sorted(eigenvector)
                print self.name
                plt.figure()
                plt.plot(sortedEigenvector)
                plt.xlabel("Rank of Eigenvector")
                plt.ylabel("Values of Eigenvector")
                plt.title("Second Eigenvector")
                plt.savefig(experiment.out_path(self.safe_name + "time=" + str(i) + ".png", "Eigenvectors"))
                plt.close()

class MoreEigenVectorsWRTFriends(MoreEigenVectors):
    def measure(self, experiment, network, iterations):
        result = getEigenVectorEigenValue(network, network.userArticleFriendGraph, iterations)
        util.writeCSV(experiment.out_path("adjacencyMatrix_iterations=" + str(iterations) + self.safe_name), result[2])


#number of common articles between users
class CommonArticles(Metric):

    def __init__(self, politicalness1=-2, politicalness2=2):
        self.politicalness1 = politicalness1
        self.politicalness2 = politicalness2

    def measure(self, experiment, network, iterations):
        userArticleGraph = network.userArticleGraph
        negativeTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness1)
        posTwo = network.getUserIdsWithSpecificPoliticalness(self.politicalness2)
        negativeTwo = random.sample(negativeTwo, min(40, len(negativeTwo)))
        posTwo = random.sample(posTwo, min(40, len(posTwo)))
        commonNeighs = []
        
        for s in posTwo:
            for v in negativeTwo:
                Nbrs = snap.TIntV()
                snap.GetCmnNbrs(userArticleGraph, s, v, Nbrs)
                commonNeighs.append(len(Nbrs))
        return mean(commonNeighs)

    def plot(self, experiment, network, history):
        plt.figure()
        print self.name
        plt.plot(history)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Common Neighbors")
        plt.title("Common Neighbors between " + str(self.politicalness1) + " and " + str(self.politicalness2))
        plt.savefig(experiment.out_path(self.safe_name + "politicalness=" + str(self.politicalness1) + " and " + str(self.politicalness2) + ".png"))
        plt.close()


    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("CommonArticles_" + "politicalness=" + str(self.politicalness1) + " and " + str(self.politicalness2)), history)


class VisualizeGraph(Metric):

    def measure(self, experiment, network, iterations):
        eigenvector, dictionary, matrix = getEigenVectorEigenValue(network)
        
        twoEigenVectors = eigenvector[1:3,:]
        #pdb.set_trace()
        #print twoEigenVectors
        self.network = network
        return (twoEigenVectors, dictionary, matrix)

    def plot(self, experiment, network, history):
        counter = 0
        for (eigenVectors, dictionary, matrix) in history:
            plt.figure()
            print self.name

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
            for politicalness in range(-2, 3):
                userIds = self.network.getUserIdsWithSpecificPoliticalness(politicalness)
                for uId in userIds:
                    mId = dictionary[uId]
                    plt.scatter(eigenVectors[mId,0], eigenVectors[mId, 1], pch[politicalness])

            for row in range(0, len(matrix)):
                for col in range(0, len(matrix[row])):
                    if matrix[row][col] == 1:
                        plt.plot([eigenVectors[row,0], eigenVectors[col, 0]], [eigenVectors[row,1], eigenVectors[col, 1]], 'k-')

            plt.title("Graph Representation")
            plt.savefig(experiment.out_path(self.safe_name + "time=" + str(counter) +".png"))
            plt.close()
            counter = counter + 1

    def save(self, experiment, history):
        util.writeCSV(experiment.out_path("eigenvectorsgraphrepresentation"), history)


class UserUserGraphCutMinimization(Metric):
    def measure(self, experiment, network, iterations):
        pass

    def plot(self, experiment, network, history):
        # Try to cluster into two clusters
        G = network.getUserUserGraphMatrix()
        L_normed = laplacian(G, normed=True)
        w, v = eigsh(L_normed, k=2, which='SM')
        print 'eigvalues:', w
        assignments = (v[:, 1] > 0)

        # Count distribution of political preference in each cluster
        countsA = collections.Counter()
        countsB = collections.Counter()
        for userId in xrange(len(assignments)):
            user = network.users.get(userId)
            if user is None:
                continue
            if assignments[userId] > 0:
                countsA[user.politicalness] += 1
            else:
                countsB[user.politicalness] += 1

        # Display counts
        print 'Cluster A:', countsA
        print 'Cluster B:', countsB



