import snap
import network


class Evaluation(object):

	def getDistribution(network):
		userArticleGraph = network.userArticleGraph
		distribution = {}
		for user in network.userList:
			nodeUserId = user.getUserId()
			userPolticalness = user.getUserPolticalness()
			for article in nodeUserId.GetOutEdges():
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

	def mean(numbers):
		return float(sum(numbers)) / max(len(numbers), 1)

	def pathsBetween2Polticalnesses(network, polticalness1=-2, polticalness2=2):
		userArticleGraph = network.userArticleGraph
		negativeTwo = network.getUserIdsWithSpecificPoltiicalness(polticalness1)
		posTwo = network.getUserIdsWithSpecificPoltiicalness(polticalness2)

		distance = []
		for user1 in negativeTwo:
			for user2 in posTwo:
				distance.append(snap.GetShortestPath(userArticleGraph, user1, user2))
		return mean(distance)

	def modularity(network):
		Nodes = snap.TIntV()
		for nodeId in network.userArticleGraph.Nodes():
			Nodes.Add(nodeId)
		return snap.getModularity(network.userArticleGraph, Nodes)

	def betweeness(network):

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
