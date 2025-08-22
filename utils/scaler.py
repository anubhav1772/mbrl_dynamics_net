## code adopted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/utils/scaler.py
import numpy as np
import os.path as path
import torch

class StandardScaler(object):
    def __init__(self, mu=None, std=None, name=None):
        self.mu = mu
        self.std = std
        self.name = name

    def fit(self, data):
        self.mu = np.mean(data, axis=0, keepdims=True)
        self.std = np.std(data, axis=0, keepdims=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        return self.std * data + self.mu
    
    def save_scaler(self, save_path):
        if self.name: # self.name is not None and self.name != ""
            mu_path = path.join(save_path, f"mu_{name}.npy")
            std_path = path.join(save_path, f"std_{name}.npy")
        else:
            mu_path = path.join(save_path, "mu.npy")
            std_path = path.join(save_path, "std.npy")
            
        np.save(mu_path, self.mu)
        np.save(std_path, self.std)
    
    def load_scaler(self, load_path):
        if self.name: # self.name is not None and self.name != ""
            mu_path = path.join(load_path, f"mu_{name}.npy")
            std_path = path.join(load_path, f"std_{name}.npy")
        else:
            mu_path = path.join(load_path, "mu.npy")
            std_path = path.join(load_path, "std.npy")
            
        self.mu = np.load(mu_path)
        self.std = np.load(std_path)

    def save_scaler_combined(self, save_path):
        fname = f"scaler_{self.name}.npz" if self.name else "scaler.npz"
        np.savez(path.join(save_path, fname), mu=self.mu, std=self.std)

    def load_scaler_combined(self, load_path):
        fname = f"scaler_{self.name}.npz" if self.name else "scaler.npz"
        data = np.load(path.join(load_path, fname))
        self.mu, self.std = data["mu"], data["std"]

    def transform_tensor(self, data: torch.Tensor):
        device = data.device
        data = self.transform(data.cpu().numpy())
        data = torch.tensor(data, device=device)
        return data