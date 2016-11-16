"""Recommendation Engines
"""

class Recommender(object):
    def makeRecommendations(self, network, reader):
        raise NotImplementedError


class RandomRecommender(object):
    def makeRecommendations(self, network, reader):
        # FIXME: only get articles
        return network.friendGraph.GetRndNId()


class CollaborativeFiltering(Recommender):
    def makeRecommendations(self, network, reader):
        raise NotImplementedError
