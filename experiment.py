import util

class Experiment(object):

	SOURCES = ["NYTimes", "WSJ", "Fox"]
	WEIGHTS_SOURCES = [1.0/3, 1.0/3, 1.0/3]

	def __init__(self):
    	self.articleGenerators = []
    	self.articleGenerators.append(ArticleGenerators(SOURCES[0], [.1, .3, 0, .3, .1]))
    	self.articleGenerators.append(ArticleGenerators(SOURCES[1], [0, .2, .5, .3, 0]))
    	self.articleGenerators.append(ArticleGenerators(SOURCES[2], [.7, .2, .1, 0, 0]))
    	self.network = Network()

    def createArticle(self):
    	idx = util.generatePoliticalness(WEIGHTS_SOURCES)
    	articleGen = self.articleGenerators[idx]
    	return articleGen.createArticle()

    def simulate(self):
    	article = createArticle()
    	readers = network.getNextReaders()
    	#recommend to readers
    	#see if readers like
    	
