import random

class User(object):

	AVERAGE_TIME_TO_LIVE = 100

	def __init__(self, politicalness, counter):
		self.politicalness = politicalness
		self.readingRate = random.expovariate(self.AVERAGE_TIME_TO_LIVE)
		self.userId = counter

	def getPoliticalness(self):
		return self.politicalness

	def getReadingRate(self):
		return self.readingRate

	def getUserId(self):
		return self.userId

	def __str__(self):
		result = "(UserId = " + str(self.userId) + ", ReadingRate = " + str(self.readingRate)
		result += ", polticalness = " + str(self.polticalness) + ")"
		return result