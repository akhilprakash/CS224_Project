import numpy as np

from article import Article


class ArticleGenerator(object):
    def __init__(self, source, distribution):
        self.source = source
        self.distribution = distribution

    def createArticle(self):
        return Article(self.source, np.random.choice([-2, -1, 0, 1, 2], p=self.distribution))
