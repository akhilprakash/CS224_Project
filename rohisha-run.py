import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import evaluation
from experiment import Experiment
from util import ensure_path_exists


# TODO: CHANGE THIS TO REFLECT WHAT YOU'RE TESTING
combined_plots_dir = ensure_path_exists(os.path.join('out', 'propagationinit.uniformplike'))


# Create metrics
idh = evaluation.ItemDegreeHeterogeneity()
nos = evaluation.NumberOfSquares()
std = evaluation.Statistics()

metrics = [
    idh,
    nos,
    std,
    evaluation.StackedBarChart()
]

# Shared parameters across experiments
constant_kwargs = {
    'numIterations': 100,
    'networkInitType': 'propagation',
    'pLikeMethod': 'uniform',
    'numOnlinePerIteration': 500,
    'numRecsPerIteration': 20,
    'metrics': metrics,
}

# Initialize experiments
exps = OrderedDict([
    ('Random', Experiment(
        recommender='Random',
        **constant_kwargs
    )),
    ('Popular', Experiment(
        recommender='Popular',
        **constant_kwargs
    )),
    ('Content-Based', Experiment(
        recommender='ContentBased',
        **constant_kwargs
    )),
    ('Friend-Based w/ Random Default', Experiment(
        recommender='InstagramWithRandomDefault',
        **constant_kwargs
    )),
    ('Friend-Based w/ Content-Based Default', Experiment(
        recommender='InstagramWithContentBasedDefault',
        **constant_kwargs
    )),
    ('Collaborative Filtering', Experiment(
        recommender='CollaborativeFiltering',
        **constant_kwargs
    )),
])

# Run experiments and save usual results
for name, exp in exps.items():
    print "Running " + name
    exp.run()
    exp.saveResults()

# Plot IDH
plt.figure()
for name, exp in exps.items():
    plt.plot(exp.histories[idh])
plt.title('Evolution of Item Degree Heterogeneity')
plt.xlabel('iterations')
plt.ylabel('item degree heterogeneity')
plt.legend(exps.keys(), loc='lower right')
plt.savefig(os.path.join(combined_plots_dir, 'idh.png'))

# Plot NoS
plt.figure()
for name, exp in exps.items():
    plt.plot(exp.histories[nos])
plt.title('Evolution of Number of Squares')
plt.xlabel('iterations')
plt.ylabel('number of squares')
plt.legend(exps.keys(), loc='upper left')
plt.savefig(os.path.join(combined_plots_dir, 'nos.png'))

# Plot std
plt.figure()
for name, exp in exps.items():
    plt.plot(exp.histories[std])
plt.title('Evolution of Readership Standard Deviation')
plt.xlabel('iterations')
plt.ylabel('average std of political orientations of an article\'s readers')
plt.legend(exps.keys(), loc='lower right')
plt.savefig(os.path.join(combined_plots_dir, 'std.png'))

