import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        # Folder containing all .hdf5 files
        self.folder_path = "mbrl_dynamics_net/dataset/PreprocessedDataset/train"

        # Define feature slices (based on your table)
        self.feature_slices = {
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

    def get_dataset_states(self):
        """Load and concatenate all states from HDF5 files in the folder."""
        all_states = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".hdf5"):
                with h5py.File(os.path.join(self.folder_path, filename), "r") as f:
                    if "states" in f:
                        states = np.array(f["states"])[:, :58]  # first 58 dims
                        all_states.append(states)
        if len(all_states) == 0:
            raise ValueError("No valid HDF5 files with 'states' dataset found.")
        return np.concatenate(all_states, axis=0)

    def plot_features(self, save_folder=None, max_cols=3):
        """Plot all features defined in feature_slices."""
        all_states = self.get_dataset_states()
        n_feats = len(self.feature_slices)
        n_rows = int(np.ceil(n_feats / max_cols))

        fig, axes = plt.subplots(n_rows, max_cols, figsize=(18, 4 * n_rows))
        axes = axes.flatten()

        for i, (feat_name, feat_slice) in enumerate(self.feature_slices.items()):
            ax = axes[i]
            values = all_states[:, feat_slice]

            if values.shape[1] == 1:
                ax.plot(values, label=feat_name)
            else:
                for j in range(values.shape[1]):
                    ax.plot(values[:, j], label=f"{feat_name}_{j+1}", alpha=0.7)

            ax.set_title(feat_name)
            ax.set_xlabel("Timestep")
            ax.legend(fontsize=8)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "all_features.png")
            plt.savefig(save_path)
            print(f"Saved feature dashboard to {save_path}")
        else:
            plt.show()

if __name__ == '__main__':
    plotter = Plotter()
    # plotter.plot_features(save_folder="mbrl_dynamics_net/plots")
    plotter.plot_features() 
