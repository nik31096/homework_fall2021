from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer, ReplayBufferAtari
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from cs285.agents.base_agent import BaseAgent
from cs285.infrastructure.pytorch_util import build_backbone_cnn


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        if agent_params.get('env_wrappers', None) is not None:
            self.env = agent_params['env_wrappers'](self.env)
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.advantage_mode = self.agent_params['advantage_mode']
        self._lambda = self.agent_params.get('lambda', None)
        self.n_step = self.agent_params.get('n_step', None)
        self.standardize_advantages = self.agent_params['standardize_advantages']
        # self.reward_normalizer = Normalizer(shape=(1,), numpy=True)

        if self.agent_params['img_based']:
            self.feature_size = 512
            backbone = build_backbone_cnn(activation='relu', feature_size=self.feature_size)
        else:
            backbone = None

        self.actor = MLPPolicyAC(
            ac_dim=self.agent_params['ac_dim'],
            ob_dim=self.agent_params['ob_dim'],
            n_layers=self.agent_params['n_layers'],
            size=self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            entropy_coeff=self.agent_params['entropy_coeff'],
            img_based=self.agent_params['img_based'],
            backbone=backbone,
            feature_size=self.feature_size
        )

        self.critic = BootstrappedContinuousCritic(
            self.agent_params,
            backbone=backbone,
            feature_size=self.feature_size
        )
        self.replay_buffer = ReplayBuffer()
        # if self.agent_params['img_based']:
        #     self.replay_buffer = ReplayBufferAtari(frame_history_len=1)
        # else:
        #     self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # self.reward_normalizer.update(re_n)
        # re_n = self.reward_normalizer.normalize(re_n)
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.critic.update(ob_no, ac_na, re_n, next_ob_no, terminal_n)

        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss = self.actor.update(ob_no, ac_na, advantage)

        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        values = self.critic.forward_np(ob_no)
        next_values = self.critic.forward_np(next_ob_no)
        if self.advantage_mode == 'td(0)':
            adv_n = re_n + self.gamma * (1 - terminal_n) * next_values - values

        elif self.advantage_mode == 'gae':
            adv_n = np.zeros((ob_no.shape[0] + 1))
            for t in reversed(range(ob_no.shape[0])):
                delta = re_n[t] + self.gamma * (1 - terminal_n[t]) * next_values[t] - values[t]
                adv_n[t] = delta + self.gamma*self._lambda*adv_n[t+1]
            adv_n = adv_n[:-1]

        elif self.advantage_mode == 'n-step return':
            # adv_n = np.zeros((ob_no.shape[0] + 1))
            # for t in range(ob_no.shape[0]):
            raise NotImplementedError("Finish implementation!!")

        else:
            raise ValueError("No such method for advantage estimation is supported! "
                             "Supported options are: 'td(0)', 'gae', 'n-step return'.")

        if self.standardize_advantages:
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)

        return adv_n

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)
