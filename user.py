import random


class User(object):
    READING_RATE = 1.0 / 100

    def __init__(self, politicalness, counter):
        self.politicalness = politicalness
        self.readingRate = random.expovariate(self.READING_RATE)
        self.userId = counter

    def getPoliticalness(self):
        return self.politicalness

    def getReadingRate(self):
        return self.readingRate

    def getUserId(self):
        return self.userId

    def setPoliticalness(self, p):
        self.politicalness = p

    def __str__(self):
        result = "(UserId = " + str(self.userId) + ", ReadingRate = " + str(
            self.readingRate)
        result += ", politicalness = " + str(self.politicalness) + ")"
        return result
