"""Recommendation Engines
"""
import heapq
import collections


# TODO: don't recommend dead articles

class Recommender(object):
    def makeRecommendations(self, network, readers, N=1):
        """
        Make recommendations for the given readers.

        :param network: Network
        :param readers: list of Users
        :param N: the number of recommendations to make for each reader
        :return: dict mapping the given readers' user IDs to a list of Articles
                 to recommend to that reader
        """
        raise NotImplementedError


class RandomRecommender(Recommender):
    def makeRecommendations(self, network, readers, N=1):
        return {r.getUserId(): network.getRandomArticles(N) for r in readers}


class PopularRecommender(Recommender):
    def makeRecommendations(self, network, readers, N=1):
        # Every is recommended the same most popular articles
        popular = heapq.nlargest(N, network.articles.itervalues(),
                                 key=lambda a: network.userArticleGraph.GetNI(a.getArticleId()).GetDeg())
        return {r.getUserId(): popular for r in readers}


class RecommendBasedOnFriends(Recommender):
    #reocmmend what is most poopular based on friends
    def makeRecommendations(self, network, readers, N=1):
        recommendation = {}
        for r in readers:
            friendsOfR = network.friendGraph.GetNI(r.getUserId()).GetOutEdges()
            articles = collections.defaultdict(int)
            for friend in friendsOfR:
                articleIds = network.articlesReadByUser(friend)
                for aId in articleIds:
                    articles[aId] = articles[aId] + 1
            sort = sorted(articles.items(), key = lambda x: x[1], reverse = True)
            recommend = sort[0:min(len(sort), N)]
            article = []
            for aid, _ in recommend:
                article.append(network.getArticle(aid))
            recommendation[r.getUserId()] = article
        return recommendation


class CollaborativeFiltering(Recommender):
    """Item-item collaborative filtering"""
    def makeRecommendations(self, network, readers, N=1):
        # (Compute similarities between all pairs of articles?)
        # For each reader:
        #   1. Compute similarity scores with all articles that the reader's friends has liked.
        #      The similarity score should be based on: the jaccard similarity of the item vectors in the ratings matrix.
        #      (note that the ratings matrix is a collapsed form of the adjacency matrix which is possible since the graph is bipartite
        #   2. Return the top N most similar articles
        raise NotImplementedError
