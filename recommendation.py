"""
Recommendation Engines

Notes: Don't recommend dead articles and or articles already liked by reader.
"""
from __future__ import division
import heapq
import itertools
import random
import math
import warnings

import numpy as np

from util import PairsDict
import util


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


def clamp(N, limit):
    if N > limit:
        warnings.warn('too few unliked articles to recommend (%d > %d)' % (N, limit),
                      RuntimeWarning)
        return limit
    else:
        return N


class Random(Recommender):
    """
    Recommend articles chosen uniformly from all the articles.
    """
    def makeRecommendations(self, network, readers, N=1):
        recs = {}
        for r in readers:
            candidates = list(network.candidateArticlesForUser(r.userId))
            recs[r.userId] = random.sample(candidates, clamp(N, len(candidates)))
        return recs


class Popular(Recommender):
    """
    Recommend globally popular articles.
    """
    def makeRecommendations(self, network, readers, N=1):
        def numLikes(article):
            return network.userArticleGraph.GetNI(article.getArticleId()).GetDeg()
        recs = {}
        for r in readers:
            candidates = list(network.candidateArticlesForUser(r.userId))
            recs[r.userId] = heapq.nlargest(clamp(N, len(candidates)), candidates, numLikes)
        return recs


class ContentBased(Recommender):
    """
    Recommends based on content qualities of the article and the reader.

    Article features:
        - Source

    User features:
        - Political leaning

    Content-based score is based on the trust?
    """
    TRUST = util.load_trust_data()

    def makeRecommendations(self, network, readers, N=1):
        recs = {}
        for reader in readers:
            def trustScore(candidate):
                return self.TRUST[candidate.source][reader.politicalness]
            candidates = list(network.candidateArticlesForUser(reader.userId))
            recs[reader.userId] = heapq.nlargest(clamp(N, len(candidates)), candidates, trustScore)
        return recs


class Instagram(Recommender):
    """
    Uniformly shows reader all the articles that their friends liked.
    """
    def __init__(self, default_recommender):
        self.default_recommender = default_recommender

    def makeRecommendations(self, network, readers, N=1):
        recs = {}
        for reader in readers:
            candidates = [
                article
                for friend in network.friendGraph.GetNI(reader.userId).GetOutEdges()
                for article in network.articlesLikedByUser(friend)
                if not network.userArticleGraph.IsEdge(article.articleId, reader.userId)
                and not article.isDead
            ]
            # If there aren't enough candidates use default recommender
            numDefault = max(0, N - len(candidates))
            sampled = random.sample(candidates, N - numDefault)
            defaulted = self.default_recommender.makeRecommendations(
                network, [reader], numDefault)[reader.userId]
            sampled.extend(defaulted)
            recs[reader.userId] = sampled
        return recs


class InstagramWithContentBasedDefault(Instagram):
    """Convenience class for Instagram recommender with fallback to ContentBased."""
    def __init__(self):
        Instagram.__init__(self, default_recommender=ContentBased())


class InstagramWithRandomDefault(Instagram):
    """Convenience class for Instagram recommender with fallback to Random."""
    def __init__(self):
        Instagram.__init__(self, default_recommender=Random())


# FIXME: make this look less like Random??
class CollaborativeFiltering(Recommender):
    """
    Item-item collaborative filtering.
    CF over Facebook likes.

    Notes on CF for binary, positive-only ratings:
    http://www.slideshare.net/koeverstrep/tutorial-bpocf

    (this is technically latent semantic analysis??)

    Edge cases:
     - When articles don't have enough likes to compute similarity,
       we default to a similarity based on their sources.
       EDIT: we default to zero similarity for these article pairs,
       and leave these articles to be recommended by our null recommender.
     - When a reader hasn't liked enough articles to compute scores for candidate
       articles, we default to content-based recommendations.
    """

    TRUST = util.load_trust_data()
    TRUST_VEC = {
        source: np.asarray([trust[-2], trust[-1], trust[0], trust[+1], trust[+2]])
        for source, trust in TRUST.iteritems()
    }

    SOURCE_SIM_WEIGHT = 1.0

    def source_similarity(self, sourceA, sourceB):
        # Cosine similarity of the source trust distributions
        # (A . B) / (||A|| ||B||)
        vecA = self.TRUST_VEC[sourceA]
        vecB = self.TRUST_VEC[sourceB]
        return vecA.dot(vecB) / math.sqrt(vecA.dot(vecA) * vecB.dot(vecB))

    def makeRecommendations(self, network, readers, N=1):
        # Compute similarities between all unique pairs of articles O(n^2)
        sim = PairsDict()
        # num_undefined = 0
        # num_zero_intersection = 0
        # nice = []
        for articleA, articleB in itertools.combinations(network.getArticles(), 2):
            a = articleA.articleId
            b = articleB.articleId
            ratersA = set(network.userArticleGraph.GetNI(a).GetOutEdges())
            ratersB = set(network.userArticleGraph.GetNI(b).GetOutEdges())

            # Modified Jaccard similarity, |A ^ B| / min(|A|, |B|)
            # This makes sure that the similarity between articles with vastly different
            # number of likes will be normalized with respect to the smaller of the
            # two liker sets.
            top = len(ratersA & ratersB)
            bot = min(len(ratersA), len(ratersB))
            sim[a, b] = top / bot if bot > 0 else 0

        # nice.sort()
        # median = nice[len(nice) // 2]
        # print 'undefined:', num_undefined, 'nonoverlapping:', num_zero_intersection, 'goodmedian:', median

        # random.shuffle(readers)
        # For each reader:
        recs = {}
        for reader in readers:
            # If there aren't enough candidates use default recommender
            candidates = list(network.candidateArticlesForUser(reader.userId))
            numDefault = max(0, N - len(candidates))

            # Compute dot product between the user's rating vector and the
            # item-item similarity vector for each candidate article. For each
            # candidate article, this is basically the sum of the similarities
            # between the candidate article and the articles that the reader has
            # liked. This will sum up exactly as many similarities as the number
            # of articles that the reader has liked. Then we should choose the
            # articles with the highest score.
            likedArticles = list(network.articlesLikedByUser(reader.userId))
            def score(candidate):
                return sum(sim[candidate.articleId, liked.articleId] for liked in likedArticles)
            thisRecs = heapq.nlargest(N - numDefault, candidates, score)

            # Fill our remaining recommendations with Content-Based default
            defaulted = ContentBased().makeRecommendations(network, [reader], numDefault)[reader.userId]
            thisRecs.extend(defaulted)
            recs[reader.userId] = thisRecs

        # DEBUG: print out recommendations for last user
        # deb = [(article.articleId, score(article)) for article in candidates]
        # deb.sort(reverse=True, key=lambda p: p[1])
        # print 'numliked:', len(likedArticles), deb

        return recs


class LFA(Recommender):
    """Latent Factor Analysis (Netflix winner style)"""
    pass
