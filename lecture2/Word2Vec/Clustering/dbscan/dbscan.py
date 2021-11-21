import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


#prepare: read vector data
data_df = pd.read_csv('../../Word2Vec2d.csv', sep=",", usecols=[0,1,2])
vectors_arr = data_df[data_df.columns[:2]].values

# Compute DBSCAN
db = DBSCAN(eps=0.03).fit(vectors_arr)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Transparent black used for noise.
        col = [0, 0, 0, 0]

    class_member_mask = (labels == k)

    xy = vectors_arr[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k')

    xy = vectors_arr[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k')

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.savefig('cluster.png')
plt.show()