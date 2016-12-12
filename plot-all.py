import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

from util import ensure_path_exists


os.chdir('out')
plot_dir = 'combined'
ensure_path_exists(plot_dir)

paths = [path for path in os.listdir('.')
         if os.path.isdir(path)
         and len(path.split('.')) == 4]


# Recommender to Color
recommenders = [
    ('RandomRecommender', 'b', 'Random'),
    ('PopularRecommender', 'g', 'Popular'),
    ('ContentBasedRecommender', 'r', 'Content-Based'),
    ('InstagramWithRandomDefaultRecommender', 'c', 'Friend-Based w/ Random Default'),
    ('InstagramWithContentBasedDefaultRecommender', 'm', 'Friend-Based w/ Content-Based Default'),
    ('CollaborativeFilteringRecommender', 'y', 'Collaborative Filtering'),
]

# Network init types
networkinits = [
    ('PropagationPoliticalPreference', ':', 'Propagated'),
    ('RandomPoliticalPreference', '-', 'Random'),
]

# PLike
plikes = [
    ('UniformPLike', '-', 'Uniform'),
    ('EmpiricalPLike', '--', 'Empirical'),
    ('IndividualPLike', ':', 'Individual'),
]

rec_legend_handles = [
    mlines.Line2D([], [], color=color, linestyle='-', label='%s Recommender' % label)
    for _, color, label in recommenders
]


# Load data
data = {}
for path in paths:
    recommender, networkinit, plike, _ = path.split('.')
    csvpath = os.path.join(path, 'statistics.csv')
    with open(csvpath, 'rb') as csvfile:
        stds = np.array([float(value) for value in csvfile])
    data[recommender, networkinit, plike] = stds


# Plot across plikes
plt.figure()
for recommender, color, reclabel in recommenders:
    for plike, style, plikelabel in plikes:
        plt.plot(data[recommender, 'RandomPoliticalPreference', plike], color+style)
legend_handles = [
    mlines.Line2D([], [], color='black', linestyle=style, label='%s PLike' % label)
    for _, style, label in plikes
]
legend1 = plt.legend(handles=legend_handles, loc='lower left', fontsize='small')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(handles=rec_legend_handles, loc='lower right', fontsize='small')

plt.title('Evolution of Readership Standard Deviation with Random Political Preferences')
plt.xlabel('iterations')
plt.ylabel('average std of political orientations of an article\'s readers')

plt.savefig(os.path.join(plot_dir, 'compare-plikes-randominit.png'))


# Plot across plikes again
plt.figure()
for recommender, color, reclabel in recommenders:
    for plike, style, plikelabel in plikes:
        plt.plot(data[recommender, 'PropagationPoliticalPreference', plike], color+style)
legend_handles = [
    mlines.Line2D([], [], color='black', linestyle=style, label='%s PLike' % label)
    for _, style, label in plikes
    ]
legend1 = plt.legend(handles=legend_handles, loc='lower left', fontsize='small')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(handles=rec_legend_handles, loc='lower right', fontsize='x-small')

plt.title('Evolution of Readership Standard Deviation with Propagated Political Preferences')
plt.xlabel('iterations')
plt.ylabel('average std of political orientations of an article\'s readers')

plt.savefig(os.path.join(plot_dir, 'compare-plikes-propagationinit.png'))


# Compare across inits
plt.figure()
for recommender, color, reclabel in recommenders:
    for networkinit, style, networkinitlabel in networkinits:
        plt.plot(data[recommender, networkinit, 'EmpiricalPLike'], color+style)
legend_handles = [
    mlines.Line2D([], [], color='black', linestyle=style, label='%s Political Preferences' % label)
    for _, style, label in networkinits
    ]
legend1 = plt.legend(handles=legend_handles, loc='lower left', fontsize='small')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(handles=rec_legend_handles, loc='lower right', fontsize='small')

plt.title('Evolution of Readership Standard Deviation with Empirical PLike')
plt.xlabel('iterations')
plt.ylabel('average std of political orientations of an article\'s readers')
plt.savefig(os.path.join(plot_dir, 'compare-polinits-empiricalplike.png'))
