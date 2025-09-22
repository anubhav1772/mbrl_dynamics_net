# import isaacgym
# assert isaacgym
import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict

import os
import h5py
from reward_aliengo_new import reward_aliengo

# from aliengo_gym.envs.wrappers.history_wrapper import HistoryWrapper
# from aliengo_gym.envs.aliengo.velocity_tracking import VelocityTrackingEasyEnv
# from config_loader import get_configs
from mbrl_dynamics_net.utils.config_loader import get_configs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class OfflineDatasetLoader:
    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self, data_load_path: str, preprocess=False):
        '''Load Real World Data 
        '''
        buffer_save_path = os.path.join(data_load_path +'/aliengo_offline_data.pt')
        # print(buffer_save_path)

        if not os.path.exists(buffer_save_path):
            if preprocess:
                dataset_real = self.load_file_preprocess(data_load_path)
            else:
                dataset_real = self.load_file(data_load_path)
            torch.save(dataset_real, buffer_save_path)
        else:
            print('Real dataset exists, loading...')
            dataset_real = torch.load(buffer_save_path)

        print('real dataset: max: {}, min: {}, mean: {}, std: {}'.format(np.max(dataset_real['rewards']), \
                                                                           np.min(dataset_real['rewards']), \
                                                                           np.mean(dataset_real['rewards']), \
                                                                           np.std(dataset_real['rewards'])))
        return dataset_real

    
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

    def compute_reward(self, observations, actions):
        obs = self.process_data(observations)
        rewards = []
        reward_calculator = reward_aliengo(action_dim=12)
        for i in range(len(obs)-1):
            reward_calculator.load_data(obs[i], actions[i])
            reward = reward_calculator.calculate_reward()
            rewards.append(reward)
        return rewards

    def load_file_preprocess(self, path: str, traj_length: float = 1000):
        dataset = {'observations': [], 'next_observations': [], 'actions': [], 'terminals': [], 'rewards': [], 'timeouts': []}
        file_id = 0
        max_reward = -1.0
        for filename in os.listdir(path):
            print(filename)
            if filename.endswith(".hdf5"):
                with h5py.File(os.path.join(path, filename), 'r') as file:
                    actions = np.array(file['actions'])
                    actions = actions[:, :12]
                    observations = np.array(file['states'])
                    # print(len(observations))    # num of steps in an episode
                    # print(len(observations[0])) # 76 (robot states - 58, camera obs - 18)

                    # np.savetxt(r'test_{}'.format(str(filename)), [np.max(actions,axis=0), np.min(actions, axis=0),
                    #                                                 np.mean(actions, axis=0), np.std(actions, axis=0)], fmt='%.3f')
                    
                    # while computing reward, we consider all 76 obss
                    reward = self.compute_reward(observations, actions)
                    max_episode_reward = np.max(reward, axis=0)
                    if(max_reward < max_episode_reward):
                        max_reward = max_episode_reward
                    print('max: {}, min: {}, mean: {}, std: {}'.format(np.max(reward,axis=0), np.min(reward, axis=0), np.mean(reward, axis=0), np.std(reward, axis=0)))

                    terminals = [False]*(len(observations)-1) + [True]
                    # dy_length = 0
                    for i in range(len(actions)-1):
                        # dy_length += 1
                        dataset['actions'].append(actions[i])
                        dataset['observations'].append(observations[:, :58][i])
                        dataset['terminals'].append(terminals[i])
                        dataset['rewards'].append(reward[i])
                        # Assign next observation (handling last timestep case)
                        dataset['next_observations'].append(observations[:, :58][i+1] if i < len(actions) - 1 else observations[:, :58][i])
                        # if dy_length % traj_length == 0:
                        #     dataset['timeouts'].append(True)
                        #     dy_length = 0
                        # else:
                        #     dataset['timeouts'].append(False)
        
        dataset['actions'] = np.array(dataset['actions'])
        dataset['observations'] = np.array(dataset['observations'])
        dataset['next_observations'] = np.array(dataset['next_observations'])
        dataset['terminals'] = np.array(dataset['terminals'])
        dataset['rewards'] = np.array(dataset['rewards']) #/ max_reward  # normalizing reward
        return dataset


    def load_file(self, path: str):
        dataset = {'observations': [], 'next_observations': [], 'actions': [], 'terminals': [], 'rewards': [], 'timeouts': []}
        file_id = 0
        max_reward = -1.0
        for files in os.listdir(path):
            print(files)
            for filename in os.listdir(os.path.join(path, files)):
                print(filename)
                if filename.endswith(".hdf5"):

                    with h5py.File(os.path.join(path, files, filename), 'r') as file:
                        actions = np.array(file['actions'])
                        actions = actions[:, :12]
                        observations = np.array(file['states'])
                        # print(len(observations))    # num of steps in an episode
                        # print(len(observations[0])) # 76

                        # np.savetxt(r'test_{}'.format(str(filename)), [np.max(actions,axis=0), np.min(actions, axis=0),
                        #                                                 np.mean(actions, axis=0), np.std(actions, axis=0)], fmt='%.3f')
                        
                        reward = self.compute_reward(observations, actions)
                        max_episode_reward = np.max(reward, axis=0)
                        if(max_reward < max_episode_reward):
                            max_reward = max_episode_reward
                        print('max: {}, min: {}, mean: {}, std: {}'.format(np.max(reward,axis=0), np.min(reward, axis=0), np.mean(reward, axis=0), np.std(reward, axis=0)))

                        terminals = [False]*(len(observations)-1) + [True]

                        for i in range(len(actions)-1):
                            dataset['actions'].append(actions[i])
                            dataset['observations'].append(observations[:, :58][i])
                            dataset['terminals'].append(terminals[i])
                            dataset['rewards'].append(reward[i])
                            # Assign next observation (handling last timestep case)
                            dataset['next_observations'].append(observations[:, :58][i+1] if i < len(actions) - 1 else observations[:, :58][i])

        dataset['actions'] = np.array(dataset['actions'])
        dataset['observations'] = np.array(dataset['observations'])
        dataset['next_observations'] = np.array(dataset['next_observations'])
        dataset['terminals'] = np.array(dataset['terminals'])
        dataset['rewards'] = np.array(dataset['rewards'])# / max_reward  # normalizing reward
        return dataset

#code adopted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/buffer/buffer.py
class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size, obs_dim), dtype=np.float32)
        self.next_observations = np.zeros((self._max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)
        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=np.float32)
        next_observations = np.array(dataset["next_observations"], dtype=np.float32)
        actions = np.array(dataset["actions"], dtype=np.float32)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        return {
            "observations": torch.from_numpy(self.observations[batch_indexes]).to(self.device),
            "actions": torch.from_numpy(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.from_numpy(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.from_numpy(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.from_numpy(self.rewards[batch_indexes]).to(self.device)
        }

if __name__ == '__main__':
    action_dim = 12
    obs_dim = 58                                 # env.get_observations()['obs'].shape[1]

    # data_load_path = 'dataset/aliengo/all'
    data_load_path = 'mbrl_dynamics_net/dataset/PreprocessedDataset/train'

    dataset_loader = OfflineDatasetLoader()
    data = dataset_loader.get_dataset(data_load_path, preprocess=True)
    for key, value in data.items():
    	print(f"{key}: {np.array(value).shape}")

    # env = VelocityTrackingEasyEnv(sim_device='cuda:0', headless=True, cfg=get_configs())
    # env = HistoryWrapper(env)

    # buffer = ReplayBuffer(int(1e6), obs_dim, action_dim)
    # buffer.load_dataset(data)
    # samples = buffer.sample(64)
    # for key in samples.keys():
    #     print(f"{key}: {samples[key].shape}")

    # print(samples.keys())
    # print(samples['observations'].shape)

