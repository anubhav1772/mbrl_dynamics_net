import os
import sys

# Add the parent folder of `mbrl_dynamics_net` to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional, Callable
# from utils import Logger, StandardScaler

from mbrl_dynamics_net.utils.logger import Logger, make_log_dirs
from mbrl_dynamics_net.utils.scaler import StandardScaler

from torch.utils.tensorboard import SummaryWriter

class Swish(nn.Module):
    '''A smooth, non-linear activation function.
    '''
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x

def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    '''A differentiable version of clamping - keeps logvar (log-variance)
    predictions within a certain range while preserving gradients.
    '''
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

# code adpoted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/nets/ensemble_linear.py
class EnsembleLinear(nn.Module):
    '''A fully-connected linear layer, but implemented as a stack of num_ensemble independent layers.
    Each ensemble member has its own weights and biases.
    Shapes:
        weight: (num_ensemble, input_dim, output_dim)
        bias: (num_ensemble, 1, output_dim)
    Uses torch.einsum for batched matrix multiplication.
    '''
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0 # L2 regularization coefficient (applies to all ensemble members)
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        # weights and biases for each ensemble member
        # shape weight: (num_ensemble, input_dim, output_dim)
        # shape bias: (num_ensemble, 1, output_dim) ->  broadcastable with batch
        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias
        if len(x.shape) == 2:
            # multiplies inputs with weights for each ensemble member
            # adds the corresponding bias — broadcasted to match batch size
            # computes a batched matrix multiplication between x and weight
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)
        x = x + bias
        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss

# code adopted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/modules/dynamics_module.py
class EnsembleDynamicsModel(nn.Module):
    '''A feedforward ensemble neural network with:
    Multiple hidden layers using EnsembleLinear
    Output layer of size 2 * (obs_dim + reward_dim):
    - First half = predicted mean
    - Second half = predicted log variance (uncertainty)
    Learnable clamp bounds: max_logvar, min_logvar
    Tracks elite model indices (e.g., top-performing ones on holdout set)
    '''
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        num_elites: int = 5,
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = 'cpu',
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.activation = activation()
        # Each layer contributes to the final prediction, so overfitting in any layer
        # can degrade model performance. That's why weight decay is important for every layer
        # hidden layers + output layer
        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        # hidden_dims is a list of integers,
        # each representing the number of nodes (neurons) in one hidden layer
        # prepends the input layer size (obs_dim+action_dim)
        # hidden dims (58+12, 200, 200, 200, 200) => values denote number of nodes in a given hidden layer
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        # Output is twice the size of (obs_dim + reward):
        # First half = predicted mean
        # Second half = predicted log variance (for uncertainty estimation)
        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        # to prevent the predicted variance from becoming too large/small
        # these are learnable clamping bounds for the log variance output
        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )
        # Keeps track of the indices of the elite models
        # used to keep track of which models in an ensemble are currently
        # selected as the "elite" ones for decision-making or prediction/evaluation
        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )
        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)

    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))

    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs

# code adopted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/dynamics/ensemble_dynamics.py
class EnsembleDynamics:
    '''Train an ensemble of neural networks to predict environment dynamics:
    Input: current observation (obs) and action (action)
    Output: change in observation (Δobs = next_obs - obs) and reward
    '''
    def __init__(
        self,
        obs_dim:int,
        action_dim:int,
        hidden_dims: Union[Tuple[int], List[int]],
        learning_rate: float,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        num_ensemble: int = 7,
        num_elites: int = 5,  # Top-performing networks used during planning
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        device:str = 'cpu',
    ) -> None:
        self.device = torch.device(device)
        # An ensemble of neural networks (instances of EnsembleDynamicsModel)
        self.model = EnsembleDynamicsModel(
                        obs_dim, action_dim, hidden_dims,
                        num_ensemble = num_ensemble,
                        num_elites = num_elites,
                        weight_decays=weight_decays,
                        device=self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), learning_rate)
        # StandardScaler for normalizing inputs
        self.scaler = StandardScaler()
        # Function to predict if a state is terminal
        self.terminal_fn = terminal_fn

    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        '''Perform a stochastic forward pass using the ensemble model to simulate the environment.
        Predicts next states and rewards from inputs.
        Uses a randomly selected elite model to sample the final prediction.
        Sample from the predicted Gaussian distribution for each ensemble member:
            (sample = μ + σ⋅ϵ)
        Choose one sample per batch from elite models (random_elite_idxs).
        Model prediction:
        - Returns mean and logvar (log-variance), indicating aleatoric uncertainty.
        - Convert logvar to standard deviation.
        Returns (next_obs, reward, terminal, info)
        '''
        #imagine single forward step
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        # next_obs = obs + Δobs
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]

        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        # Use terminal_fn to detect termination.
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}

        return next_obs, reward, terminal, info

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        '''Converts raw environment data (dict of arrays) into:
        Input: obs + action
        Target: delta_obs + reward
        '''
        # print(next(iter(data.items())))
        # observations
        # actions
        # terminals
        # rewards
        # timeouts
        # for key, value in data.items():
        #     print(key)

        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"].reshape(-1,1)
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        wandb = None,
        tensorboard_writer = None,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01,
    ) -> None:
        '''Trains the ensemble model on a dataset using early stopping and holdout validation.
        Split data into train and holdout sets.
        Steps:
            Scale inputs.
            Initialize per-model losses.
            Train loop:
                Call learn() for training
                Evaluate on holdout using validate()
                If performance improves, save the model using model.update_save()
                If no improvement over multiple epochs, stop early
            Select elite models (lowest holdout loss)
        '''
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        # Holdout set (aka validation set) size: For monitoring validation performance (generalization) during training
        # Reserve up to 1000 points for holdout set
        # Randomly shuffle and split data into:
        # Training set & Holdout (validation) set
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        # Normalize Inputs
        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        # High initial loss for all ensemble models
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]
        # Random indices used to assign different data permutations to each ensemble model (for diversity)
        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            # Train models on current training data
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            # Validate each model on holdout data
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            if wandb is not None:
                wandb.log({'timestep': epoch,
                    'loss/dynamics_train_loss': train_loss,
                    'loss/dynamics_holdout_loss': holdout_loss,
                })
            else:
                tensorboard_writer.add_scalar("loss/dynamics_train_loss", train_loss, epoch)
                tensorboard_writer.add_scalar("loss/dynamics_holdout_loss", holdout_loss, epoch)

            # Shuffle dataset for next epoch
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            # Update best models
            # If improvement > 1%, mark the model as improved and update the stored loss.
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss

            # Save improved models
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                # If no improvement -> increment stagnation counter
                cnt += 1

            # Halts training if:
            # - No improvement for max_epochs_since_update epochs, or
            # - Epoch limit is reached
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        # Select top num_elites models with lowest holdout losses
        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))

    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)

            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)

    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss

    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
        # TODO: save loss, other metrics to json

    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)

# def rollout(init_obss: np.ndarray, rollout_length: int) -> Tuple[Dict[str, np.ndarray], Dict]:
#     num_transitions = 0
#     rewards_arr = np.array([])
#     rollout_transitions = defaultdict(list)

#     # rollout
#     observations = init_obss
#     for _ in range(rollout_length):
#         if self._uniform_rollout:
#             actions = np.random.uniform(
#                 -1,
#                 1,
#                 size=(len(observations), self.action_dim)
#             )
#         else:
#             actions = self.select_action(observations)
#         next_observations, rewards, terminals, info = self.dynamics.step(observations, actions)
#         rollout_transitions["obss"].append(observations)
#         rollout_transitions["next_obss"].append(next_observations)
#         rollout_transitions["actions"].append(actions)
#         rollout_transitions["rewards"].append(rewards)
#         rollout_transitions["terminals"].append(terminals)

#         # tracks how many total transitions (not timesteps) were collected
#         # since batch size may shrink over time after filtering terminal states
#         # num_transitions is a sum of all transitions collected across all surviving batches
#         # which is not the same as (rollout_length × initial_batch_size),
#         # because some episodes terminate early and are excluded from later steps.
#         num_transitions += len(observations)
#         rewards_arr = np.append(rewards_arr, rewards.flatten())

#         nonterm_mask = (~terminals).flatten()
#         if nonterm_mask.sum() == 0:
#             break

#         # print(observations.shape)
#         # print(next_observations.shape)
#         # print(rewards.shape)
#         # print(terminals.shape)
#         observations = next_observations[nonterm_mask]

#     for k, v in rollout_transitions.items():
#         rollout_transitions[k] = np.concatenate(v, axis=0)

#     return rollout_transitions, \
#         {"num_transitions": num_transitions, "reward_mean": rewards_arr.mean()}

def train_dynamics_model():
    import argparse
    import random
    from mbrl_dynamics_net.utils.termination_fns import get_termination_fn
    # from utils.logger import Logger, make_log_dirs
    from mbrl_dynamics_net.utils.buffer import OfflineDatasetLoader
    # from utils.scaler import StandardScaler

    from datetime import datetime
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="aliengo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", type=bool, default=True)
    parser.add_argument("--obs_dim", type=int, default=58)
    parser.add_argument("--action_dim", type=int, default=12)
    parser.add_argument("--dynamics-lr", type=float, default=3e-4)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--run-name', type=str, default=datetime.now().strftime("run_%Y%m%d-%H%M%S"), help='used for logging to distingush different runs')

    args = parser.parse_args()

    config = {
        "dynamic_module": {
            "hidden_dims": args.dynamics_hidden_dims,
            "lr": args.dynamics_lr,
            "num_ensemble": args.n_ensemble,
            "num_elites": args.n_elites,
            "weight_decay": args.dynamics_weight_decay,
            "termination_fn": args.task,
            "class": "EnsembleDynamics",
            },
        "meta": {
            "device": args.device
            }
        }

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Clear unused GPU memory
    torch.cuda.empty_cache()

    # Logger
    log_dirs = make_log_dirs(args.task, 'test/dynamics', args.seed, vars(args), run_name=args.run_name)
    print(f"log_dirs = {log_dirs}")
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    use_wandb = True
    try:
        # Attempt to initialize wandb and start tracking
        wandb.init(
            project="anubhav1772-itmo-university",
            name=f"DYN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config = config,
            resume="never", # fresh run
        )
        print("W&B initialized successfully")
    except wandb.errors.errors.CommError as e:
        # In case of an error with wandb, catch the exception and use TensorBoard instead
        print(f"W&B error occurred: {e}. \nUsing TensorBoard for logging instead.")
        use_wandb = False

        # # Generate dynamic log directory using timestamp
        # tensorboard_log_dir = os.path.join("runs", f"DYN_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard_writer = SummaryWriter(log_dir=log_dirs)

    # retrain flag provide flexibility while testing
    # when model already exists but still we want retraining,
    # we can set it to True
    # retrain = True
    dynamic_model_path = os.path.join(logger.model_dir, f"dynamics_{args.seed}.pth")
    print(dynamic_model_path)

    # Aliengo Offline Data
    data_load_path = 'dataset/PreprocessedDataset/train'
    data = OfflineDatasetLoader().get_dataset(data_load_path, preprocess=True)
    for key, value in data.items():
        print(f"{key}: {np.array(value).shape}")

    dynamics = EnsembleDynamics(
        args.obs_dim, args.action_dim,
        args.dynamics_hidden_dims,
        args.dynamics_lr,
        get_termination_fn(args.task),
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device)

    if os.path.isfile(dynamic_model_path) and args.retrain == False:
        print(f"Trained dynamics exists at {logger.model_dir}, loading...")
        dynamics.load(logger.model_dir)
        print("Load successful!!")
    else:
        # dynamic training
        print("Starting dynamics model training...")
        if use_wandb:
            dynamics.train(data, logger, wandb=wandb)
        else:
            dynamics.train(data, logger, tensorboard_writer=tensorboard_writer)
            tensorboard_writer.close()

if __name__ == '__main__':
    train_dynamics_model()