from __future__ import division

import csv
import datetime
import os
import math
import random
import sys
import time
from contextlib import contextmanager

import numpy as np


def weighted_choice(weights):
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    return np.random.choice(range(len(weights)), p=weights)


def with_prob(p):
    return random.random() < p


def writeCSV(fileName, value):
    with open(fileName + '.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(0, len(value)):
            spamwriter.writerow([value[i]])


def load_trust_data():
    """
    trust[source][pol] = percentage of users with preference `pol` that trust `source`
    """
    trust = {}
    with open(data_path('percof-readers-trust.csv')) as f:
        reader = csv.reader(f, delimiter=",")
        next(reader, None)  # skip headers
        for row in reader:
            source = row[0]
            trust[source] = {
                -2: float(row[6]),
                -1: float(row[5]),
                 0: float(row[4]),
                +1: float(row[3]),
                +2: float(row[2]),
            }
    return trust


def human_time(*args, **kwargs):
    "http://stackoverflow.com/a/34654259"
    secs  = float(datetime.timedelta(*args, **kwargs).total_seconds())
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    parts = []
    for unit, mul in units:
        if secs / mul >= 1 or mul == 1:
            if mul > 1:
                n = int(math.floor(secs / mul))
                secs -= n * mul
            else:
                n = secs if secs != int(secs) else int(secs)
            parts.append("%s %s%s" % (n, unit, "" if n == 1 else "s"))
    return ", ".join(parts)


from collections import deque
import itertools

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.izip(a, b)


class ProgressBar(object):
    def __init__(self, total, width=40, window_size=100):
        self.total = total
        self.width = width
        self.start = time.time()
        self._last_line_length = 0
        self.window_size = window_size
        self.t = deque(maxlen=2)
        self.v = deque(maxlen=window_size-1)
        self.a = deque(maxlen=window_size-2)
        self.last_ratio = 0

    def __enter__(self):
        return self

    def _seconds_left(self, ratio):
        # Update ticks
        d_ratio = ratio - self.last_ratio
        t = time.time()
        v = (t - self.t[-1]) / d_ratio if self.t else 0
        a = (v - self.v[-1]) / d_ratio if self.v else 0
        self.t.append(t)
        self.v.append(v)
        self.a.append(a)
        self.last_ratio = ratio

        # TODO: Use running update instead of iterating
        # Update moving averages
        vavg = sum(self.v) / float(len(self.v))
        aavg = sum(self.a) / float(len(self.a))

        return int(vavg + 0.5 * aavg)

        now = time.time()
        elapsed = now - self.start
        return int(elapsed / ratio * (1. - ratio))

    def update(self, done):

        # Erase
        sys.stdout.write('\r')
        sys.stdout.write(' ' * self._last_line_length)
        sys.stdout.write('\r')

        # Draw bar
        ratio = done / self.total
        bars = int(ratio * self.width)
        line = u''.join([
            u'\u2593' * bars,
            u'\u2591' * (self.width - bars),
            ' %.1f%% complete,' % (ratio * 100),
            ' about %s left' % human_time(seconds=self._seconds_left(ratio)),
        ])
        sys.stdout.write(line)
        sys.stdout.flush()

        # Save last line length
        self._last_line_length = len(line)

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

def out_path(filename, subdir=None):
    output_dir = OUTPUT_BASE_DIRECTORY
    if subdir is not None:
        output_dir = os.path.join(output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print_error('Created directory ' + output_dir)
    return os.path.join(output_dir, filename)


def ensure_path_exists(path):
    """
    Creates path if it doesn't exist yet.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print_error('Created directory ' + path)
    return path


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
    def __contains__(self, key):
        u, v = key
        if u < v:
            return dict.__contains__(self, key)
        else:
            return dict.__contains__(self, (v, u))

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


@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    stdoutfd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        os.dup2(to.fileno(), stdoutfd) # fd writes to 'to' file

    with os.fdopen(os.dup(stdoutfd), 'wb') as old_stdout:
        with open(to, 'wb') as tofile:
            _redirect_stdout(to=tofile)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different