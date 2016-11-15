import random

def generatePoliticalness(weights):
    	rand = random.random()
    	summation = sum(weights)
    	for i in range(0, len(weights)):
    		weights[i] = weights[i]/ (summation * 1.0)
    	cumsum = weights[0]
    	for i in range(0, len(weights)):
    		if rand < cumsum:
    			return i
    		cumsum = cumsum + weights[i]
    	#should not reach here
    	raise Exception("Should not reach here")