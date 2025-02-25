import numpy as np
import torch

import time
import copy


def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):
    model = models[0]
    # true
    true_states = perform_actions(env, action_sequence)['observation']

    # predicted
    ob = np.expand_dims(true_states[0], 0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac, 0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
    return np.mean((a - b)**2)


def sample_trajectory(env, policy, max_path_length, render=False, render_mode='rgb_array'):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs_frames, next_obs_frames = [], []
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        # use the most recent ob to decide what to do
        # if policy.img_based:
        #     obs.append(ob)
        #     obs_history = np.concatenate(obs[-4:], axis=2)
        #     if obs_history.shape[-1] < 4:
        #         obs_history = np.pad(obs_history, ((0, 0), (0, 0), (4 - obs_history.shape[-1], 0)))
        #     obs_frames.append(obs_history)
        #     ac = policy.get_action(obs_history)
        # else:
        #     obs.append(ob)
        #     ac = policy.get_action(ob)
        obs.append(ob)
        ac = policy.get_action(ob)

        # if len(ac.shape) > 0:
        #     print(ac.shape)
        #     ac = ac[0]
        acs.append(ac)

        # take that action and record results
        next_ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(next_ob)
        # if policy.img_based:
        #     next_obs_history = np.concatenate([obs_history[..., 1:], next_ob], axis=-1)
        #     if next_obs_history.shape[-1] < 4:
        #         next_obs_history = np.pad(next_obs_history, ((0, 0), (0, 0), (4 - next_obs_history.shape[-1], 0)))
        #     next_obs_frames.append(next_obs_history)

        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = 1 if done or steps >= max_path_length else 0  # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break

        ob = next_ob

    # if policy.img_based:
    #     return Path(obs_frames, image_obs, acs, rewards, next_obs_frames, terminals)
    # else:
    #     return Path(obs, image_obs, acs, rewards, next_obs, terminals)
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode='rgb_array'):
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.

    TODO implement this function
    Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env=env, policy=policy, max_path_length=max_path_length,
                                 render=render, render_mode=render_mode)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []
    for i in range(ntraj):
        path = sample_trajectory(env=env, policy=policy, max_path_length=max_path_length,
                                 render=render, render_mode=render_mode)
        paths.append(path)

    return paths


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs:
        image_obs = np.stack(image_obs, axis=0)

    return {"observation": np.concatenate(obs).astype(np.float32),
            "image_obs": None if not image_obs else np.concatenate(image_obs).astype(np.uint8),
            "reward": np.concatenate(rewards).astype(np.float32),
            "action": np.array(acs).astype(np.float32),
            "next_observation": np.concatenate(next_obs).astype(np.float32),
            "terminal": np.array(terminals).astype(np.float32)}


def convert_listofrollouts(paths):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]

    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards


def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data-mean)/(std+eps)


def unnormalize(data, mean, std):
    return data*std+mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp) #(num data points, dim)

    #mean of data
    mean_data = np.mean(data, axis=0)

    #if mean is 0,
    #make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    #width of normal distribution to sample noise from
    #larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (data.shape[0],)))

    return data


class Normalizer(object):
    # Taken from https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), numpy=False, device='cpu'):
        self.numpy = numpy
        if self.numpy:
            self.mean = np.zeros(shape, 'float32')
            self.var = np.ones(shape, 'float32')
        else:
            self.mean = torch.zeros(shape, dtype=torch.float32).to(device)
            self.var = torch.ones(shape, dtype=torch.float32).to(device)

        self.count = epsilon

    def normalize(self, inputs, subtract_mean=True):
        normalized_inputs = inputs / (self.var + 1e-6)
        if subtract_mean:
            normalized_inputs = normalized_inputs - self.mean

        return normalized_inputs

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.std()
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        if not self.numpy:
            delta_square = torch.square(delta)
        else:
            delta_square = np.square(delta)
        M2 = m_a + m_b + delta_square * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def save(self, path, name):
        if not self.numpy:
            torch.save(self.mean.cpu(), f'{path}/{name}_norm_mean.pt')
            torch.save(self.var.cpu(), f'{path}/{name}_norm_std.pt')
        else:
            np.save(f'{path}/{name}_norm_mean.npy', self.mean)
            np.save(f'{path}/{name}_norm_std.npy', self.var)

    def load(self, path, name):
        if not self.numpy:
            self.mean = torch.load(f'{path}/{name}_norm_mean.pt')
            self.var = torch.load(f'{path}/{name}_norm_std.pt')
        else:
            self.mean = np.load(f'{path}/{name}_norm_mean.npy')
            self.var = torch.load(f'{path}/{name}_norm_std.npy')
