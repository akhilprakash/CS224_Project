"""Recommendation Engines
"""
from __future__ import division
import heapq
import collections
import itertools

from util import PairsDict


# TODO: Uniform-among-friends-liked-articles Null Recommender
# TODO: don't recommend dead articles
# TODO: LSA-based recommender??

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
        articles = []
        for a in network.articles.itervalues():
            if not a.getIsDead():
                articles.append(a)
        popular = heapq.nlargest(N, articles,
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
                    if not network.articles[aId].getIsDead():
                        articles[aId] = articles[aId] + 1
            sort = sorted(articles.items(), key = lambda x: x[1], reverse = True)
            recommend = sort[0:min(len(sort), N)]
            article = []
            for aid, _ in recommend:
                article.append(network.getArticle(aid))
            recommendation[r.getUserId()] = article
        return recommendation


class CollaborativeFiltering(Recommender):
    """
    Item-item collaborative filtering.
    CF over Facebook likes.

    Notes on CF for binary, positive-only ratings:
    http://www.slideshare.net/koeverstrep/tutorial-bpocf

    (this is technically latent semantic analysis??)
    """
    def makeRecommendations(self, network, readers, N=1):
        # Compute similarities between all unique pairs of articles O(n^2)
        sim = PairsDict()
        for articleA, articleB in itertools.combinations(network.articles, 2):
            ratersA = set(network.userArticleGraph.GetNI(articleA).GetOutEdges())
            ratersB = set(network.userArticleGraph.GetNI(articleB).GetOutEdges())
            # TODO: make sure that new articles are handled properly here
            # so that we don't need to do anything special to initialize new
            # articles -- they should be recommend to new users.
            # Use Jaccard similarity with correction to prevent divide-by-zero
            sim[articleA, articleB] = (len(ratersA | ratersB) + 1) / (len(ratersA & ratersB) + 1)

        # For each reader:
        recs = {}
        for reader in readers:
            likedArticles = set(network.userArticleGraph.GetNI(reader.userId).GetOutEdges())
            candidateArticles = list(article for article in network.articles if article not in likedArticles)

            # Compute dot product between the user's rating vector and the item-item similarity vector
            # for each candidate article. For each candidate article, this is basically the sum of the similarities
            # between the candidate article and the articles that the reader has liked.
            # This will sum up exactly as many similarities as the number of articles that the reader has liked.
            # Then we should choose the articles with the highest score.
            def score(candidate):
                return sum(sim[candidate, liked] for liked in likedArticles)
            topN = heapq.nlargest(N, candidateArticles, score)
            recs[reader.userId] = [network.getArticle(a) for a in topN]

        return recs

