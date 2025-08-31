import os
import h5py
import numpy as np
import pandas as pd

def get_dataset_stats():
    # Folder containing all .hdf5 files
    folder_path = "mbrl_dynamics_net/dataset/PreprocessedDataset/train"

    # Define feature slices
    feature_slices = {
        "gravity_vector": slice(0, 3),
        "x_vel": slice(3, 4),
        "y_vel": slice(4, 5),
        "yaw_vel": slice(5, 6),
        "body_height": slice(6, 7),
        "step_freq": slice(7, 8),
        "gait": slice(8, 11),
        "durations": slice(11, 12),
        "footswing_height": slice(12, 13),
        "body_pitch": slice(13, 14),
        "body_roll": slice(14, 15),
        "stance_width": slice(15, 16),
        "stance_length": slice(16, 17),
        "aux_reward": slice(17, 18),
        "dof_pos": slice(18, 30),
        "dof_vel": slice(30, 42),
        "actions": slice(42, 54),
        "clock_inputs": slice(54, 58),
    }

    # Collect all states across episodes
    all_states = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".hdf5"):
            with h5py.File(os.path.join(folder_path, filename), "r") as f:
                states = np.array(f["states"])[:, :58]  # first 58 dims
                all_states.append(states)

    # Concatenate all episodes into one big array [N, 58]
    all_states = np.concatenate(all_states, axis=0)

    # Compute global mean and std for each feature group
    stats = {}
    for feat, sl in feature_slices.items():
        feat_data = all_states[:, sl]

        # Flatten in case slice has multiple dims (e.g., dof_pos, actions)
        feat_data = feat_data.reshape(-1, feat_data.shape[-1])

        stats[feat] = {
            "mean": feat_data.mean(axis=0),
            "std": feat_data.std(axis=0),
        }

    return stats

def print_constant_features():
    folder_path = "mbrl_dynamics_net/dataset/PreprocessedDataset/train"
    all_states = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".hdf5"):
            with h5py.File(os.path.join(folder_path, fname), "r") as f:
                states = np.array(f["states"])[:, :58]  # first 58 features
                all_states.append(states)

    # Concatenate across episodes
    all_states = np.concatenate(all_states, axis=0)  # [N, 58]

    # Compute global std
    feature_std = all_states.std(axis=0)

    # Threshold (to avoid float precision issues)
    threshold = 1e-8
    constant_features = np.where(feature_std < threshold)[0]

    print("Constant feature indices:", constant_features)

# print(stats)

# # Convert to a DataFrame for readability
# rows = []
# for feat, values in stats.items():
#     mean = values["mean"].ravel()  # ensure 1D
#     std = values["std"].ravel()
#     for i in range(len(mean)):
#         rows.append({
#             "feature": feat,
#             "dim": i,
#             "mean": mean[i],
#             "std": std[i]
#         })

# stats_df = pd.DataFrame(rows)

# # Show entire table (not just first 5 rows)
# pd.set_option("display.max_rows", None)
# print(stats_df)


print_constant_features()
