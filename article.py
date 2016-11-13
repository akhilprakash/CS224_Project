import random

class Article(object):

	AVERAGE_TIME_TO_LIVE = 100

    def __init__(self, source, politicalness, articleId=None):
    	self.source = source
    	self.politicalness = politicalness
    	self.timeToLive = random.expovariate(AVERAGE_TIME_TO_LIVE)
        self.articleId = articleId

    def getSource(self):
    	return self.source

    def getPolticalness(self):
    	return self.politicalness

    def getTimeToLife(self):
    	return self.timeToLive

    def getArticleId(self):
        return self.articleId

    def setArticleId(self, aId):
        self.articleId = aId