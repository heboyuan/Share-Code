import cv2
import gym
import numpy as np
from random import randint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('BreakoutDeterministic-v4').unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU")
else:
    print("CPU")

# parameters
N_ACTIONS = env.action_space.n
N_STEPS = 5
LOG_FRAME = 10000
LEARNING_RATE = 1e-4
GAMMA = 0.99
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
TD_LAMBDA = 1
MAX_GRAD_NORM = 40


def process_img(raw_rgb):
    raw_grey = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    res_grey = cv2.resize(raw_grey, dsize=(84, 110), interpolation=cv2.INTER_NEAREST)
    chop_grey = res_grey[21:105,:]
    return chop_grey

log_file = open('./A2C_log', 'w+')

def log(reward_count, episode_count, frame_count):
    episode_reward = reward_count / episode_count
    print(reward_count)
    print(episode_reward)
    log_file.write(str(episode_reward) + '\n')
    log_file.flush()
    print("========== " + str(frame_count) + " Frames ==========")


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.lin3 = nn.Linear(32*9*9, 256)
        
        self.policy = nn.Linear(256, N_ACTIONS)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x.cuda()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.lin3(x.view(x.size(0), -1)))
        
        pi = self.policy(x)
        v = self.v(x)

        return pi, v

# initialize network
AC_net = ActorCritic().to(device)
AC_net.eval()
optimizer = optim.Adam(AC_net.parameters(), lr = LEARNING_RATE)

def train():
    done = True
    log_frame = LOG_FRAME
    frame_count = 0
    reward_count = 0
    episode_count = 0

    states = [None for _ in range(4)]

    while True:
        values = []
        log_probs = []
        entropies = []
        rewards = []

        # log the information if a game is done
        if done:
            episode_count += 1
            if frame_count > log_frame:
                log(reward_count, episode_count, frame_count)
                reward_count = 0
                episode_count = 0
                log_frame += LOG_FRAME

            env.reset()
            states = [None for _ in range(4)]
            cur_life = env.ale.lives()
            new_life = env.ale.lives()
            done = False

        # initialize the states
        env.step(1)
        for i in range(4):
            states[i] = process_img(env.step(randint(0, N_ACTIONS-1))[0])/255.0

        # play N steps of game
        for step in range(N_STEPS):
            frame_count += 1

            policy, value = AC_net(torch.FloatTensor(states[:], device = device).unsqueeze(0))

            # calculate probability
            prob = F.softmax(policy, dim=1)
            log_prob = F.log_softmax(policy, dim=1)

            # calculate entropy
            entropy = -(log_prob*prob).sum(1, keepdim=True)

            # choose action and its probability
            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)

            # take action and get game state
            state, reward, done, info = env.step(action.cpu().numpy()[0][0])
            reward = np.clip(reward,-1,1)
            reward_count += reward
            state = process_img(state)/255.0
            states.pop(0)
            states.append(state)
            new_life = info["ale.lives"]

            # store the value
            values.append(value)
            entropies.append(entropy)
            log_probs.append(log_prob)
            rewards.append(reward)

            if new_life != cur_life:
                break

        # if game end, last value is 0, else make predict
        if new_life == cur_life:
            _, value = AC_net(torch.FloatTensor(states[:], device = device).unsqueeze(0))
            final_reward = value.data
        else:
            final_reward = torch.zeros(1, 1).cuda()
            cur_life = new_life
        values.append(final_reward)

        # calculate the loss
        policy_loss = 0
        value_loss = 0
        R = final_reward
        GAE = torch.zeros(1, 1).cuda()
        for i in range(len(rewards) - 1, -1 ,-1):
            R = GAMMA*R + rewards[i]
            value_loss += 0.5*(R - values[i])**2

            TD_error = GAMMA*values[i+1].data + rewards[i] - values[i].data
            GAE = GAE*GAMMA*TD_LAMBDA + TD_error
            policy_loss -= log_probs[i]*GAE - ENTROPY_COEF*entropies[i]

        loss = policy_loss + VALUE_LOSS_COEF*value_loss
        loss.backward()

        nn.utils.clip_grad_norm_(AC_net.parameters(), MAX_GRAD_NORM)
        optimizer.zero_grad()
        optimizer.step()

train()