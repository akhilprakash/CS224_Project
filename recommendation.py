"""Recommendation Engines
"""
import heapq
import collections

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
    def makeRecommendations(self, network, readers, N=1):
        raise NotImplementedError
