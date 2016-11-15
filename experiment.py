import util
import random
from network import Network
from articleGenerator import ArticleGenerator
from reccomendation import Recommendation
from evaluation import Evaluation

class Experiment(object):

    SOURCES = ["NYTimes", "WSJ", "Fox"]
    WEIGHTS_SOURCES = [1.0/3, 1.0/3, 1.0/3]
    NUM_SIMULATIONS = 100

    def __init__(self):
        self.articleGenerators = []
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[0], [.1, .3, 0, .3, .1]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[1], [0, .2, .5, .3, 0]))
        self.articleGenerators.append(ArticleGenerator(self.SOURCES[2], [.7, .2, .1, 0, 0]))
        self.network = Network()
        self.recommendation = Recommendation()
        self.distributionResults = []
        self.pathResults = []

    def createArticle(self):
        idx = util.generatePoliticalness(self.WEIGHTS_SOURCES)
        articleGen = self.articleGenerators[idx]
        return articleGen.createArticle()

    def PLike(self, reader, article):
        return .5

    def simulate(self):
        article = self.createArticle()
        readers = self.network.getNextReaders()
        self.network.addArticle(article)
        for reader in readers:
            rec = self.recommendation.makeRecommendation(self.network, reader)

        for reader in readers:
            probLike = self.PLike(reader, article)
            rand = random.random()
            if rand < probLike:
                self.network.addEdge(reader, article)

    	#recommend to readers
    	#see if readers like
    	#if it does add edge

        self.distributionResults.append(Evaluation().getDistribution(self.network))
        self.pathResults.append(Evaluation().pathsBetween2Polticalnesses(self.network))
    
    def runAllSimulation(self):
        for i in range(0, self.NUM_SIMULATIONS):
            self.simulate()
            print i
        print self.distributionResults

if __name__ == "__main__":
    exp = Experiment()
    exp.runAllSimulation()