"""Recommendation Engines
"""
from __future__ import division
import heapq
import itertools
import random

from util import PairsDict


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


class Random(Recommender):
    """
    Recommend articles chosen uniformly from all the articles.
    """
    def makeRecommendations(self, network, readers, N=1):
        return {r.getUserId(): network.getRandomArticles(N) for r in readers}


class Popular(Recommender):
    """
    Recommend the same set of globally popular articles to every reader.
    """
    def makeRecommendations(self, network, readers, N=1):
        articles = []
        for a in network.articles.itervalues():
            if not a.getIsDead():
                articles.append(a)

        def numLikes(article):
            return network.userArticleGraph.GetNI(article.getArticleId()).GetDeg()
        popular = heapq.nlargest(N, articles, numLikes)
        return {r.getUserId(): popular for r in readers}


class Instagram(Recommender):
    """
    Uniformly shows reader all the articles that their friends liked.

    Potential problems:
    shows all articles liked by friends regardless of how long ago that was.
    (this might be taken care of by the isDead check now)
    """
    def makeRecommendations(self, network, readers, N=1):
        recs = {}
        for reader in readers:
            candidates = [
                article
                for friend in network.friendGraph.GetNI(reader.userId).GetOutEdges()
                for article in network.articlesLikedByUser(friend)
            ]
            recs[reader] = random.sample(candidates, N)
        return recs


# TODO: LSA-based recommender??


# FIXME: don't recommend dead articles
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
            likedArticles = {article.articleId for article in network.articlesLikedByUser(reader)}
            candidateArticles = [
                article
                for article in network.articles.itervalues()
                if article.articleId not in likedArticles
                and not article.isDead
            ]

            # Compute dot product between the user's rating vector and the item-item similarity vector
            # for each candidate article. For each candidate article, this is basically the sum of the similarities
            # between the candidate article and the articles that the reader has liked.
            # This will sum up exactly as many similarities as the number of articles that the reader has liked.
            # Then we should choose the articles with the highest score.
            def score(candidate):
                return sum(sim[candidate.articleId, liked] for liked in likedArticles)
            recs[reader.userId] = heapq.nlargest(N, candidateArticles, score)

        return recs


class LFA(Recommender):
    """Latent Factor Analysis (Netflix winner style)"""
    pass
