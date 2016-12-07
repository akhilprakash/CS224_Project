from __future__ import division

import csv
import os
import sys
import time

import numpy as np


def weighted_choice(weights):
    weights = np.asarray(weights)
    weights /= np.sum(weights)
    return np.random.choice(range(len(weights)), p=weights)


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


DATA_BASE_DIRECTORY = 'data'
OUTPUT_BASE_DIRECTORY = 'out'


# Warning:
# This is only defined at module load time, which means that running two
# experiments in the same session without reloading this module will clobber
# the output data of the first experiment.
timestamp = time.strftime("%Y-%m-%d-%H-%M")


def out_path(filename, subfolder=None):
    """
    Returns a path for a new output file in the format:
        out/YY-MM-DD-hh-mm/[subfolder/]filename
    """
    output_dir = os.path.join(OUTPUT_BASE_DIRECTORY, timestamp)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print_error('Created directory ' + output_dir)
    if subfolder is not None:
        output_dir = os.path.join(output_dir, subfolder)
    return os.path.join(output_dir, filename)


def data_path(filename):
    return os.path.join(DATA_BASE_DIRECTORY, filename)


class PairsDict(dict):
    """
    dict with keys that are 2-tuples (pairs), such that set and get are invariant
    to the order of the pairs. Aka,

        d = PairsDict()
        d[a, b] = "hello"
        assert d[a, b] == d[b, a]
    """
    def __setitem__(self, key, value):
        u, v = key
        if u < v:
            return dict.__setitem__(self, key, value)
        else:
            return dict.__setitem__(self, (v, u), value)

    def __getitem__(self, key):
        u, v = key
        if u < v:
            return dict.__getitem__(self, key)
        else:
            return dict.__getitem__(self, (v, u))
