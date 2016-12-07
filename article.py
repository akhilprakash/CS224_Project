import random

class Article(object):

    AVERAGE_TIME_TO_LIVE = 1.0/30

    def __init__(self, source, articleId=None):
        self.source = source
        self.timeToLive = random.expovariate(self.AVERAGE_TIME_TO_LIVE)
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

    def incrementTimeToLive(self, increment):
        self.timeToLive = self.timeToLive + increment

    def __str__(self):
        result = "(Source = " + self.source
        result += ", timeToLive = " + str(self.timeToLive) + ", article Id = " + str(self.articleId)
        result += ", isDead = " + str(self.isDead) + ")"
        return result