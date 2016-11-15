import snap
import network


class Evaluation(object):

	def getDistribution(self, network):
		userArticleGraph = network.userArticleGraph
		distribution = {}
		for user in network.userList:
			nodeUserId = user.getUserId()
			userPolticalness = user.getPoliticalness()
			for article in userArticleGraph.GetNI(nodeUserId).GetOutEdges():
				articlePoliticalness = network.getArticlePolticalness(article)
				if userPolticalness in distribution:
					innerDict = distribution[userPolticalness]
					if articlePoliticalness in innerDict:
						innerDict[articlePoliticalness] = innerDict[articlePoliticalness] + 1
					else:
						innerDict[articlePoliticalness] = 1
				else:
					distribution[userPolticalness] = {articlePoliticalness : 1}
		return distribution

	def mean(self, numbers):
		return float(sum(numbers)) / max(len(numbers), 1)

	def pathsBetween2Polticalnesses(self, network, polticalness1=-2, polticalness2=2):
		userArticleGraph = network.userArticleGraph
		negativeTwo = network.getUserIdsWithSpecificPoltiicalness(polticalness1)
		posTwo = network.getUserIdsWithSpecificPoltiicalness(polticalness2)

		distance = []
		# for user1 in negativeTwo:
		# 	for user2 in posTwo:
		# 		#figure out why this is not working
		# 		#distance.append(GetShortPath(userArticleGraph, user1, user2))
		# 		x = 1
		return self.mean(distance)

	def modularity(self, network):
		Nodes = snap.TIntV()
		for nodeId in network.userArticleGraph.Nodes():
			Nodes.Add(nodeId)
		return snap.getModularity(network.userArticleGraph, Nodes)

	def betweeness(self, network):

		Nodes = snap.TIntFltH()
		Edges = snap.TIntPrFltH()
		snap.GetBetweennessCentr(network.userArticleGraph, Nodes, Edges, 1.0)
		for node in Nodes:
			print "node: %d centrality: %f" % (node, Nodes[node])
		# for edge in Edges:
  #   		print "edge: (%d, %d) centrality: %f" % (edge.GetVal1(), edge.GetVal2(), Edges[edge])

		betweenessCentr = []
		for edge in Edges:
			betweenessCentr.append((edge, Edges[edge]))

		return betweenessCentr

	def getUserDegreeDistribution(self, network):
		userArticleGraph = network.userArticleGraph
		degree = []
		for user in network.userList:
			uId = user.getUserId()
			degree.append(userArticleGraph.GetNI(uId).GetOutDeg())
		return degree

	def getArticleDegreeDistribution(self, network, str):
		userArticleGraph = network.userArticleGraph
		degree = []
		for article in network.articleList:
			aId = article.getArticleId()
			if str == "all" or (str == "alive" and not article.getIsDead()) or (str == "dead" and article.getIsDead()):
				degree.append(userArticleGraph.GetNI(aId).GetOutDeg())
		return degree

	def getDistributionOfLifeTime(self, network, iterations):
		lifeTime = []
		for article in network.articleList:
			if not article.getIsDead():
				lifeTime.append(article.getTimeToLive() - iterations)
		return lifeTime