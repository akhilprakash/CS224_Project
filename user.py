class User(object):

	AVERAGE_TIME_TO_LIVE = 100

    def __init__(self, politicalness, counter):
    	self.politicalness = politicalness
    	self.readingRate = random.expovariate(AVERAGE_TIME_TO_LIVE)
    	self.userId = counter

    def getPolticalness(self):
    	return self.politicalness

    def getReadingRate(self):
    	return self.readingRate