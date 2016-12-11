import os

import matplotlib.pyplot as plt
import numpy as np

import evaluation
from experiment import Experiment


# Create metrics
idh = evaluation.ItemDegreeHeterogeneity()
nos = evaluation.NumberOfSquares()
std = evaluation.Statistics()

metrics = [
    idh,
    nos,
    std,
]

# Shared parameters across experiments
constant_kwargs = {
    'numIterations': 100,
    'networkInitType': 'propagation',
    'pLikeMethod': 'empirical',
    'numOnlinePerIteration': 500,
    'numRecsPerIteration': 20,
    'metrics': metrics,
}

# Initialize experiments
exps = {
    'Random': Experiment(
        recommender='Random',
        **constant_kwargs
    ),
    'Content-Based': Experiment(
        recommender='ContentBased',
        **constant_kwargs
    ),
    'Instagram w/ Random Default': Experiment(
        recommender='InstagramWithRandomDefault',
        **constant_kwargs
    ),
    'Instagram w/ Content-Based Default': Experiment(
        recommender='InstagramWithContentBasedDefault',
        **constant_kwargs
    ),
    'Collaborative Filtering': Experiment(
        recommender='CollaborativeFiltering',
        **constant_kwargs
    )
}

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
plt.legend(exps.keys())
plt.savefig(os.path.join('out', '', 'all-idhs.png'))

# Plot NoS
plt.figure()
for name, exp in exps.items():
    plt.plot(exp.histories[nos])
plt.title('Evolution of Number of Squares')
plt.xlabel('iterations')
plt.ylabel('number of squares')
plt.legend(exps.keys())
plt.savefig(os.path.join('out', 'all-nos.png'))

# Plot std
plt.figure()
for name, exp in exps.items():
    plt.plot(exp.histories[std])
plt.title('Evolution of Readership Variance')
plt.xlabel('iterations')
plt.ylabel('average std of political orientations of an article\'s readers')
plt.legend(exps.keys())
plt.savefig(os.path.join('out', 'all-std.png'))

