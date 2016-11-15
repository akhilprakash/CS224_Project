import util
import random
import network
import articleGenerator
import reccomendation

class Experiment(object):

    SOURCES = ["NYTimes", "WSJ", "Fox"]
    WEIGHTS_SOURCES = [1.0/3, 1.0/3, 1.0/3]
    NUM_SIMULATIONS = 1000

    def __init__(self):
        self.articleGenerators = []
        self.articleGenerators.append(ArticleGenerators(SOURCES[0], [.1, .3, 0, .3, .1]))
        self.articleGenerators.append(ArticleGenerators(SOURCES[1], [0, .2, .5, .3, 0]))
        self.articleGenerators.append(ArticleGenerators(SOURCES[2], [.7, .2, .1, 0, 0]))
        self.network = Network()
        self.recommendation = Recommendation()
        self.distributionResults = []
        self.pathResults = []

    def createArticle(self):
        idx = util.generatePoliticalness(WEIGHTS_SOURCES)
        articleGen = self.articleGenerators[idx]
        return articleGen.createArticle()

    def PLike(reader, article):
        return .5

    def simulate(self):
        article = createArticle()
        readers = network.getNextReaders()
        for reader in readers:
            rec = self.recommendation.makeRecommendation(network, reader)

        for reader in readers:
            probLike = PLike(reader, article)
            rand = random.random()
            if rand < probLike:
                network.addEdge(reader, article)

    	#recommend to readers
    	#see if readers like
    	#if it does add edge

        self.distributionResults.append(Evaluation().getDistribution(network))
        self.pathResults.append(Evaluation().pathsBetween2Polticalnesses(network))
    
    def runAllSimulation(self):
        for _ in range(0, NUM_SIMULATIONS):
            simulate(self)

if __name__ == "__main__":
    exp = Experiment()
    exp.runAllSimulation()