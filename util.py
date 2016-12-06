from __future__ import division
import random
import csv
import sys
import time
import os


def generatePoliticalness(weights):
    	rand = random.random()
    	summation = sum(weights)
        if summation == 0:
            for i in range(0, len(weights)):
                weights[i] = 1.0/len(weights)
        else:
        	for i in range(0, len(weights)):
        		weights[i] = weights[i]/ (summation * 1.0)
    	cumsum = 0
    	for i in range(0, len(weights)):
            cumsum = cumsum + weights[i]
            if rand < cumsum:
                return i
    		
    	#should not reach here
    	raise Exception("Should not reach here")


def writeCSV(fileName, value):
    with open(fileName + '.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(value)):
            spamwriter.writerow([value[i]])


class ProgressBar(object):
    def __init__(self, total=None, width=40, use_newlines=False):
        self.total = total
        self.width = width
        self.use_newlines = use_newlines
        self.start = time.time()

    def __enter__(self):
        return self

    def _seconds_left(self, ratio):
        now = time.time()
        elapsed = now - self.start
        return elapsed / ratio * (1. - ratio)

    def update(self, done):
        ratio = done / self.total
        bars = int(ratio * self.width)
        sys.stdout.write('\r[')
        sys.stdout.write('|' * bars)
        sys.stdout.write(' ' * (self.width - bars))
        sys.stdout.write('] %.1f%% complete, about %d min left' %
                         (ratio * 100, self._seconds_left(ratio) / 60.))
        if self.use_newlines:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.update(self.total)
        sys.stdout.write('\n')


def visual_xrange(stop, **kwargs):
    with ProgressBar(total=stop, **kwargs) as progress:
        for i in xrange(stop):
            yield i
            progress.update(i+1)


def print_error(s):
    print >>sys.stderr, s


def data_path(filename):
    return os.path.join('data', filename)

now = time.strftime("%c")
def out_path(filename, subfolder=""):
    if subfolder == "":
        if not os.path.exists('out ' + str(now).replace(":", "")):
            print_error('Created `out` directory for result files.')
            os.mkdir('out ' + str(now).replace(":", ""))

        return os.path.join('out ' + str(now).replace(":", ""), filename)
    else:
        if not os.path.exists('out ' + str(now).replace(":", "") + "/" + subfolder):
            print_error('Created out' + subfolder +' directory for result files.')
            os.makedirs('out ' + str(now).replace(":", "")+ "/" + subfolder)

        return os.path.join('out ' + str(now).replace(":", "")+ "/" + subfolder, filename)
