import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(
            self,
            ac_dim,
            ob_dim,
            n_layers,
            size,
            discrete=False,
            learning_rate=1e-4,
            training=True,
            nn_baseline=False,
            entropy_coeff=1e-2,
            img_based=False,
            backbone=None,
            feature_size=512,
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
        self.entropy_coeff = entropy_coeff
        self.img_based = img_based

        if self.discrete:
            if not self.img_based:
                self.logits_na = ptu.build_mlp(
                    input_size=self.ob_dim,
                    output_size=self.ac_dim,
                    n_layers=self.n_layers,
                    size=self.size
                )
            else:
                self.logits_na = ptu.build_cnn(
                    output_size=self.ac_dim,
                    activation='leaky_relu',
                    backbone=backbone,
                    feature_size=feature_size
                )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(), lr=self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size
            )
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                lr=self.learning_rate
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from hw1 or hw2
        if len(obs.shape) == 3:
            obs = obs[np.newaxis, ...]
        obs = ptu.from_numpy(obs)
        dist = self.forward(obs)
        action = dist.sample().cpu().data.numpy()

        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from hw1 or hw2
        if self.discrete:
            logits = self.logits_na(observation)
            probs = torch.softmax(logits, dim=-1)
            action_distribution = distributions.Categorical(probs=probs)
        else:
            # mean = self.mean_net(observation)
            # action_distribution = distributions.Normal(loc=mean, scale=self.logstd.exp())
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(batch_mean, scale_tril=batch_scale_tril)

        return action_distribution


class MLPPolicyAC(MLPPolicy):
    def update(self, observations, actions, adv_n=None):
        assert adv_n is not None, "Can't do actor update without advantage function"
        if not isinstance(observations, torch.Tensor):
            observations = ptu.from_numpy(observations)

        if not isinstance(actions, torch.Tensor):
            actions = ptu.from_numpy(actions)

        if not isinstance(adv_n, torch.Tensor):
            adv_n = ptu.from_numpy(adv_n).detach()

        action_dist = self.forward(observations)
        log_probs = action_dist.log_prob(actions)
        loss = -torch.mean(log_probs*adv_n + self.entropy_coeff*action_dist.entropy())

        self.optimizer.zero_grad()
        loss.backward()
        if self.discrete:
            nn.utils.clip_grad_norm_(self.logits_na.parameters(), 0.5)
        else:
            nn.utils.clip_grad_norm_(self.mean_net.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.logstd, 0.5)
        self.optimizer.step()

        return loss.item()
