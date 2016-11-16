"""Recommendation Engines
"""

class Recommender(object):
    def makeRecommendations(self, network, reader):
        raise NotImplementedError


class RandomRecommender(object):
    def makeRecommendations(self, network, reader):
        network.friendGraph.GetRndNId()


class CollaborativeFiltering(Recommender):
    def makeRecommendations(self, network, reader):
        raise NotImplementedError
