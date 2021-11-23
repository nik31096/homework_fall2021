from cs285.policies.MLP_policy import MLPPolicySL

import gym
import torch


env = gym.make("Ant-v2")
state = env.reset()

policy = MLPPolicySL(env.action_space.shape[0], env.observation_space.shape[0], n_layers=2, size=64)
policy.load_state_dict(torch.load("../../data/q1_bc_ant_Ant-v2_09-11-2021_16-14-04/policy_itr_0.pt"))

reward_sum = 0
while True:
    action = policy.get_action(state)
    next_state, reward, done, info = env.step(action)
    reward_sum += reward

    # env.render(mode='human')

    if done:
        break

    state = next_state

print("Reward sum:", reward_sum)
