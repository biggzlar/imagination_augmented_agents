import gym
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

# setup paths for local imports
sys.path.append(os.getcwd())

from envs import make_env
from networks import EnvModel, I3A
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

print(os.getcwd())
# create environment
env_ = make_env('Frostbite-v0', 0, 0)()
# env_ = SubprocVecEnv([env_])

# load environment model
env_path = os.getcwd() + '/exp/env.pth'
env_model = EnvModel(num_channels=7)
if env_path:
    model_params = torch.load(env_path)
    env_model.load_state_dict(model_params['model'])
    print('Environment model loaded.')

# load agent
model_path = os.getcwd() + '/exp/epoch-20.pth'
model = I3A(env_model=env_model, actions=env_.action_space.n)
if os.path.isfile(model_path):
    snapshot = torch.load(model_path)
    model.load_state_dict(snapshot['model'])
    print('Agent model loaded.')
else:
    raise FileNotFoundError('no snapshot found at "{0}"'.format(model_path))


# play episodes 
n_episodes = 1000
for _ in range(n_episodes):
    state = env_.reset()
    model.eval()
    score = 0
    hx, cx = Variable(torch.zeros(1, 1, model.state_size)), Variable(torch.zeros(1, 1, model.state_size))
    mask = Variable(torch.zeros(1, 1, 1))
    while True:
        env_.render()
        state = Variable(torch.from_numpy(state).unsqueeze(0).float())

        # Action logits
        action_logit = model.forward((state, (hx, cx), mask))[1]
        action_probs = F.softmax(action_logit)
        log_probs = F.log_softmax(action_logit)
        actions = action_probs.multinomial()
        action = actions[0, 0]

        # visualize????
        state, reward, done, _ = env_.step(action.data[0])
        score += reward
        if done:
            break

    print(' * Actual Score of {}'.format(score))