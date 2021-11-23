import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 device='cpu',
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline
        self.device = device

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(self.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(self.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=self.device)
            )
            self.logstd.to(self.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            obs = obs
        else:
            obs = obs[None]

        d = self.forward(obs)
        action = ptu.to_numpy(d.sample())

        return action

    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        obs = ptu.from_numpy(observation).to(self.device)
        action_mean = self.mean_net(obs)
        d = distributions.Normal(loc=action_mean, scale=self.logstd.exp())

        return d


class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        d = self.forward(observations)
        pred_actions = d.rsample()
        if not isinstance(actions, torch.Tensor):
            actions = ptu.from_numpy(actions)
        loss = self.loss(pred_actions, actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
