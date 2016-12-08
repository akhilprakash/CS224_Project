import random

class Article(object):
    AVERAGE_LIFETIME_EXPERIMENT_RATIO = 0.333  # articles live for a third of the simulation

    def __init__(self, source, numIterations, t, articleId=None):
        average_ttl = self.AVERAGE_LIFETIME_EXPERIMENT_RATIO * numIterations
        self.source = source
        self.timeToLive = t + random.expovariate(lambd=1. / average_ttl)
        self.articleId = articleId
        self.isDead = False

    def getSource(self):
        return self.source

    def getTimeToLive(self):
        return self.timeToLive

    def getArticleId(self):
        return self.articleId

    def setArticleId(self, aId):
        self.articleId = aId

    def getIsDead(self):
        return self.isDead

    def setIsDead(self, isDead):
        self.isDead = isDead

    def __str__(self):
        result = "(Source = " + self.source
        result += ", timeToLive = " + str(self.timeToLive) + ", article Id = " + str(self.articleId)
        result += ", isDead = " + str(self.isDead) + ")"
        return result