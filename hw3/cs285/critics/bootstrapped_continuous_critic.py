import torch

from .base_critic import BaseCritic
from torch import nn
from torch import optim

from cs285.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
    Notes on notation:

    Prefixes and suffixes:
    ob - observation
    ac - action
    _no - this tensor should have shape (batch self.size /n/, observation dim)
    _na - this tensor should have shape (batch self.size /n/, action dim)
    _n  - this tensor should have shape (batch self.size /n/)

    Note: batch self.size /n/ is defined at runtime.
    is None
    """
    def __init__(self, hparams, backbone=None, feature_size=512):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['size']
        self.n_layers = hparams['n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.num_target_updates = hparams['num_target_updates']
        self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        if not hparams['img_based']:
            self.critic_network = ptu.build_mlp(
                self.ob_dim,
                1,
                n_layers=self.n_layers,
                size=self.size,
            )
        else:
            self.critic_network = ptu.build_cnn(
                output_size=1,
                activation='relu',
                backbone=backbone,
                feature_size=feature_size
            )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

    def forward(self, obs):
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs):
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no, ac_na, reward_n, next_ob_no, terminal_n):
        """
        Update the parameters of the critic.

        let sum_of_path_lengths be the sum of the lengths of the paths sampled from Agent.sample_trajectories
        let num_paths be the number of paths sampled from Agent.sample_trajectories

        arguments:
            ob_no: shape: (sum_of_path_lengths, ob_dim)
            ac_na: length: sum_of_path_lengths. The action taken at the current step.
            reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                the reward for each timestep
            next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
            terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                at that timestep of 0 if the episode did not end

        returns:
            training loss
        """
        # TODO: Implement the pseudocode below: do the following (
        #  self.num_grad_steps_per_target_update * self.num_target_updates)
        #  times:
        #  every self.num_grad_steps_per_target_update steps (which includes the
        #  first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        #  every time, update this critic using the observations and targets
        #
        # HINT: don't forget to use terminal_n to cut off the V(s') (ie set it
        #       to 0) when a terminal state is reached
        # HINT: make sure to squeeze the output of the critic_network to ensure
        #       that its dimensions match the reward
        if not isinstance(ob_no, torch.Tensor):
            ob_no = ptu.from_numpy(ob_no)

        if not isinstance(next_ob_no, torch.Tensor):
            next_ob_no = ptu.from_numpy(next_ob_no)

        if not isinstance(reward_n, torch.Tensor):
            reward_n = ptu.from_numpy(reward_n)

        if not isinstance(terminal_n, torch.Tensor):
            terminal_n = ptu.from_numpy(terminal_n)

        for target_update in range(self.num_target_updates):
            with torch.no_grad():
                next_values = self(next_ob_no)
                targets = reward_n + self.gamma * (1 - terminal_n) * next_values

            for step in range(self.num_grad_steps_per_target_update):
                values = self(ob_no)
                loss = self.loss(values, targets)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
                self.optimizer.step()

        return loss.item()
