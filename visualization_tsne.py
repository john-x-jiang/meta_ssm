import os
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import argparse
from utils import Params

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def parse_args():
    """
    Args:
        config: json file with hyperparams and exp settings
        seed: random seed value
        stage: 1 for traing VAE, 2 for optimization,  and 12 for both
        logging: 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='b01', help='config filename')
    parser.add_argument('--test', type=int, default=1, help='config filename')
    args = parser.parse_args()
    return args

args = parse_args()

# filename of the params
fname_config = args.config + '.json'
# read the params file
json_path = os.path.join(os.path.dirname(os.path.realpath('__file__')), "config", fname_config)
hparams = Params(json_path)
exp_dir = os.path.join(os.path.dirname(os.path.realpath('__file__')), 'experiments', hparams.exp_name, hparams.exp_id)

# Load in the embeddings given the three physics datasets
if args.test == 1:
    zs, labels = [], []
    for i in [0, 4, 6, 9, 12, 14]:
    # for i in [6, 9, 12, 14]:
        data = np.load('{}/embeddings/embeddings_qry_{}.npz'.format(exp_dir, i))
        zs.append(data['c_zs'])
        labels.append(data['labels'])

    # zs = np.vstack((data1['c_mus'], data2['c_mus'], data3['c_mus']))
    zs = np.vstack(zs)
    labels = np.vstack(labels)
elif args.test == 2:
    zs, labels = [], []
    for i in [0, 6, 12]:
    # for i in [9, 14]:
        data = np.load('{}/embeddings/embeddings_qry_{}.npz'.format(exp_dir, i))
        zs.append(data['c_zs'])
        labels.append(data['labels'])

    zs = np.vstack(zs)
    labels = np.vstack(labels)
elif args.test == 3:
    zs, labels = [], []
    for i in [4, 9, 14]:
    # for i in [9, 14]:
        data = np.load('{}/embeddings/embeddings_qry_{}.npz'.format(exp_dir, i))
        zs.append(data['c_zs'])
        labels.append(data['labels'])

    zs = np.vstack(zs)
    labels = np.vstack(labels)
elif args.test == 0:
    zs, labels = [], []
    for i in range(15):
    # for i in range(5, 15):
        data = np.load('{}/embeddings/embeddings_qry_{}.npz'.format(exp_dir, i))
        zs.append(data['c_zs'])
        labels.append(data['labels'])
    zs = np.vstack(zs)
    labels = np.vstack(labels)

print(zs.shape, labels.shape)
print(np.unique(labels))

# Take a K-Means algorithm applied to the embedding space
n_clusters = 3 # 9
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(zs)
# np.savez('{}/embeddings/kmeans_clusters.npz'.format(exp_dir), centers=kmeans.cluster_centers_, labels=kmeans.labels_)

# Take a TSNE of both the embeddings and cluster centers to get a visualizable lower-dim space
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=2000, early_exaggeration=12, verbose=True)
fitted = tsne.fit(np.vstack((kmeans.cluster_centers_, zs)))
print("Finished after {} iterations".format(fitted.n_iter))
tsne_embedding = fitted.embedding_
print(len(kmeans.labels_))

embeds = tsne_embedding[n_clusters:]
centroids = tsne_embedding[:n_clusters]

# Plot them together on a color-coated and shared plot
plt.figure(figsize=(8,6))
np.random.seed(123)
colors = np.random.choice(list(matplotlib.colors.CSS4_COLORS), 15, replace=False)
# legends = ["Bouncing Ball", "Pendulum", "Mass Spring"]
legends = np.arange(0, len(np.unique(labels)))
for oidx, idx in enumerate(np.unique(labels)):
    subset = embeds[np.where(labels == idx)[0]]
    # plt.scatter(subset[:, 0], subset[:, 1], label=legends[oidx], c=colors[int(idx)])
    if int(idx) in [1, 2, 3]:
        plt.scatter(subset[:, 0], subset[:, 1], label='t_0', c=colors[1])
    elif int(idx) in [5, 7, 8]:
        plt.scatter(subset[:, 0], subset[:, 1], label='t_1', c=colors[5])
    elif int(idx) in [10, 11, 13]:
        plt.scatter(subset[:, 0], subset[:, 1], label='t_2', c=colors[11])
    else:
        plt.scatter(subset[:, 0], subset[:, 1], label=int(idx), c=colors[int(idx)])

plt.scatter(centroids[:, 0], centroids[:, 1], label='Centroids', c='k', marker='x')
plt.title("TSNE Plot of Context Cs Embeddings over the Dynamics")
plt.legend()
# plt.show()
plt.savefig('{}/embeddings/TSNE_{}.png'.format(exp_dir, args.test), format='png', bbox_inches='tight')


# Assign all of the embeddings to each cluster and save cluster centers for initial-state-reconstruction
def euclid_dist(x, y):
    x = np.reshape(x, [1, x.shape[0]])
    y = np.reshape(y, [y.shape[0], -1])
    return np.sqrt(np.sum((x - y)**2, axis=1))


center_labels = []
for centroid in kmeans.cluster_centers_:
    distances = euclid_dist(centroid, zs)
    indices = labels[np.argsort(distances)]

    label = int(mode(indices[:1000])[0][0])
    center_labels.append(label)

print(center_labels)

sort = np.argsort(center_labels)
centroids_sorted = kmeans.cluster_centers_[sort]
center_labels = np.array(center_labels)[sort]

np.savez('{}/embeddings/kmeans_clusters.npz'.format(exp_dir), centers=centroids_sorted, labels=center_labels)
