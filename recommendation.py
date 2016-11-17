"""Recommendation Engines
"""
import heapq


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


class CollaborativeFiltering(Recommender):
    def makeRecommendations(self, network, readers, N=1):
        raise NotImplementedError
