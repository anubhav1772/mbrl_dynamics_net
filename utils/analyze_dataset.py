import matplotlib.pyplot as plt
from buffer import OfflineDatasetLoader

import os
import numpy as np
from typing import Optional, Union, Tuple, Dict

import os
import h5py
from reward_aliengo_new import reward_aliengo

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# data = OfflineDatasetLoader().get_dataset('dataset/go1')

def process_data(self, observations):
        # print(f"Processing...")

        x0 = observations[0, 58]
        y0 = observations[0, 59]
        z0 = observations[0, 60]
        r0 = observations[0, 67]
        p0 = observations[0, 68]
        theta0 = observations[0, 69]

        N = len(observations)
        L = 0.16

        observations[:, 58] -= x0 * np.ones(N)
        observations[:, 59] -= y0 * np.ones(N)
        observations[:, 60] -= z0 * np.ones(N)
        observations[:, 67] -= r0 * np.ones(N)
        observations[:, 68] -= p0 * np.ones(N)
        observations[:, 69] -= theta0 * np.ones(N)

        for i in range(len(observations)):
            C = np.cos(observations[i, 69] + theta0)
            S = np.sin(observations[i, 69] + theta0)

            observations[i, 58] = observations[i, 58] - L * np.cos(theta0) + L
            observations[i, 59] = observations[i, 59] - L * np.sin(theta0)

            observations[i, 61] += L * S * observations[i, 72]
            observations[i, 62] -= L * C * observations[i, 72]

            R = np.array([[C, S, 0],
                          [-S, C, 0],
                          [0, 0, 1]])
            
            v_ref = np.array([[observations[i, 61]],
                              [observations[i, 62]],
                              [0]])
            # Rotate + bias
            v = np.dot(R, v_ref) + np.array([[0], [L * observations[i, 72]], [0]]) 
            observations[i, 61] = v[0]
            observations[i, 62] = v[1]

        return observations

def compute_reward(observations, actions):
    obs = process_data(observations=observations)
    rewards = []
    reward_calculator = reward_aliengo(action_dim=12)
    for i in range(len(obs)-1):
        reward_calculator.load_data(obs[i], actions[i]) #reward_calculator.load_data(obs[i+1], actions[i]) use next obs to calculate reward
        reward = reward_calculator.calculate_reward()
        rewards.append(reward)
    return rewards

dataset = {'observations': [], 'next_observations': [], 'actions': [], 'terminals': [], 'rewards': [], 'timeouts': []}
file_id = 0
path = 'mbrl_dynamics_net/dataset/PreprocessedDataset/train'
rewards_per_episode = []
obs = []
# for files in os.listdir(path):
#     print(files)
#     for filename in os.listdir(os.path.join(path, files)):
#         print(filename)
#         if filename.endswith(".hdf5"):

#             with h5py.File(os.path.join(path, files, filename), 'r') as file:
#                 actions = np.array(file['actions'])
#                 # print(actions.shape)
#                 actions = actions[:, :12]
#                 observations = np.array(file['states'])
#                 # print(observations.shape)
#                 obs.append(observations)
#                 #print(len(observations[0])) # 76
#                 #np.savetxt(r'test_{}'.format(str(filename)), [np.max(actions,axis=0), np.min(actions, axis=0),
#                 #                                                np.mean(actions, axis=0), np.std(actions, axis=0)], fmt='%.3f')
#                 reward = compute_reward(observations, actions)
#                 rewards_per_episode.extend(reward)

for filename in os.listdir(path):
    print(filename)
    if filename.endswith(".hdf5"):
        with h5py.File(os.path.join(path, filename), 'r') as file:
            actions = np.array(file['actions'])
            # print(actions.shape)
            actions = actions[:, :12]
            observations = np.array(file['states'])
            # print(observations.shape)
            obs.append(observations)
            #print(len(observations[0])) # 76
            #np.savetxt(r'test_{}'.format(str(filename)), [np.max(actions,axis=0), np.min(actions, axis=0),
            #                                                np.mean(actions, axis=0), np.std(actions, axis=0)], fmt='%.3f')
            max_reward = -1
            reward = compute_reward(observations, actions)
            max_reward = max(max_reward, max(reward))
            rewards_per_episode.extend(reward)    

# for key, value in data.items():
#     if key == 'rewards':
#         print(f"{key}: {value}")

# for ep in data:
#     print(ep['rewards'])

# Plot reward per episode
# rewards_per_episode = [sum(ep['rewards']) for ep in data]
plt.hist(np.array(rewards_per_episode)/max_reward, bins=25)
plt.title("Reward Distribution")
plt.xlabel("Episode Return")
plt.ylabel("Count")
plt.show()

# # Concatenate and sample
# states = np.concatenate(obs, axis=0)
# n_sample = min(5000, len(states))
# indices = np.random.choice(len(states), size=n_sample, replace=False)
# states_sampled = states[indices]
#
# # Normalize
# states_sampled = StandardScaler().fit_transform(states_sampled)
#
# # t-SNE projection to 2D
# tsne_proj = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(states_sampled)
#
# import hdbscan
# clusterer = hdbscan.HDBSCAN(min_cluster_size=20).fit(tsne_proj)
# labels = clusterer.labels_
#
# plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=labels, cmap='tab10', s=4, alpha=0.6)
# plt.title("HDBSCAN Clustering on t-SNE")
# plt.show()


# DBSCAN clustering on 2D t-SNE output
# dbscan = DBSCAN(eps=1.5, min_samples=5).fit(tsne_proj)
# labels = dbscan.labels_  # -1 means "noise" (not in any cluster)
#
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=labels, cmap='tab10', s=5, alpha=0.6)
# plt.title("t-SNE + DBSCAN Clustering of Aliengo States")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.colorbar(label="Cluster ID (-1 = noise)")
# plt.show()

# Plot
# plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], alpha=0.3, s=2)
# plt.title("State Diversity (t-SNE)")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.show()

# KMeans Clustering
# kmeans = KMeans(n_clusters=15, random_state=0).fit(tsne_proj)
# labels = kmeans.labels_  # shape: (5000,)

# plt.figure(figsize=(7, 6))
# plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
# plt.title("t-SNE Clustering of States")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.colorbar(label='Cluster ID')
# plt.show()


# states = np.concatenate(obs, axis=0)
# tsne_proj = TSNE(n_components=2).fit_transform(states)
#
# plt.scatter(tsne_proj[:,0], tsne_proj[:,1], alpha=0.2, s=1)
# plt.title("State Diversity (t-SNE)")
# plt.show()
