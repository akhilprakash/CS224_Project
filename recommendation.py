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
    """Item-item collaborative filtering"""
    K = 5

    def makeRecommendations(self, network, readers, N=1):
        # 1. Compute similarities between all unique pairs of articles O(n^2)
        sim = PairsDict()
        for articleA, articleB in itertools.combinations(network.articles, 2):
            ratersA = set(network.userArticleGraph.GetNI(articleA).GetOutEdges())
            ratersB = set(network.userArticleGraph.GetNI(articleB).GetOutEdges())
            # TODO: make sure that new articles are handled properly here
            # so that we don't need to do anything special to initialize new
            # articles -- they should be recommend to new users.
            # Use Jaccard similarity with correction to prevent divide-by-zero
            sim[articleA, articleB] = (len(ratersA | ratersB) + 1) / (len(ratersA & ratersB) + 1)

        # 2. Compute k-nearest neighbors to the articles liked by the readers
        # Collect the set of articles liked by the given readers
        likedArticles = set(
            article
            for reader in readers
            for article in network.userArticleGraph.GetNI(reader.userId).GetOutEdges()
        )
        # For each article find KNNs

        # TODO: DO THIS INSTEAD
        # compute dot product between the user's rating vector and the item-item similarity vector
        # for each candidate article. For each candidate article, this is basically the sum of the similarities
        # between the candidate article and the articles that the reader has liked.
        # Then we should choose the articles with the highest score.
        # This will sum up exactly as many similarities as the number of articles that the reader has liked.
        # (this is technically latent semantic analysis??)
        # http://www.slideshare.net/koeverstrep/tutorial-bpocf

        # For each reader:
        for reader in readers:
            likedArticles = set(network.userArticleGraph.GetNI(reader.userId).GetOutEdges())
            candidateArticles = list(article for article in network.articles if article not in likedArticles)

            # 2. For each article, compute its k-nearest neighbors in terms of
            #    similarity among the articles read by the reader
            knns = {}
            for article in candidateArticles:
                knns[article] = heapq.nlargest(likedArticles, lambda other: sim[article, other])

            # 3. Since we are working only with binary ratings (i.e. likes),
            #    we don't need to "estimate" the rating of the candidate articles,
            #    since the the k-nearest neighbors are already the recommendations
            # TODO: limit to just the articles that the reader's friends has liked
            # TODO: use baseline estimates to de-bias
            estPLike = {}
            for article in candidateArticles:
                estPLike = sum()


            # 4. Recommend the articles with the top N estimated ratings for each
            #    reader.

        raise NotImplementedError
