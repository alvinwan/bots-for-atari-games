"""
Step 5 - Fully featured deep q-learning network.
"""

import cv2
import gym
import numpy as np
import random
import tensorflow as tf
from step_4_a3c import a3c_model
random.seed(0)  # make results reproducible
tf.set_random_seed(0)

num_episodes = 10


def downsample(state):
    return cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)[None]


def main():
    env = gym.make('SpaceInvaders-v0')  # create the game
    env.seed(0)  # make results reproducible
    rewards = []

    model = a3c_model(load='models/SpaceInvaders-v0.tfmodel')
    for _ in range(num_episodes):
        episode_reward = 0
        states = [downsample(env.reset())]
        while True:
            if len(states) < 4:
                action = env.action_space.sample()
            else:
                frames = np.concatenate(states[-4:], axis=3)
                action = np.argmax(model([frames]))
            state, reward, done, _ = env.step(action)
            states.append(downsample(state))
            episode_reward += reward
            if done:
                print('Reward: %d' % episode_reward)
                rewards.append(episode_reward)
                break
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))


if __name__ == '__main__':
    main()


