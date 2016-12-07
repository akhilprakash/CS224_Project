import util
from article import Article

class ArticleGenerator(object):

    def __init__(self, source, distribution):
        self.source = source
        self.distribution = distribution

    def getSource(self):
        return self.source

    def createArticle(self):
        indexToPoliticalness = {0: -2, 1: -1, 2:0, 3:1, 4:2}
        index = util.generatePoliticalness(self.distribution)
        politicalness = indexToPoliticalness[index]
        #somehow figure out ids
        return Article(self.source, politicalness)
