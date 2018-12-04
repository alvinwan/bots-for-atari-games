"""
Bot 6 - Fully featured deep q-learning network.
"""

import argparse
import cv2
import gym
import numpy as np
import random
import tensorflow as tf
from bot_6_a3c import a3c_model
random.seed(0)  # make results reproducible
tf.set_random_seed(0)

num_episodes = 10


def downsample(state):
    return cv2.resize(state, (84, 84), interpolation=cv2.INTER_LINEAR)[None]


def main():
    parser = argparse.ArgumentParser(description='Run DQN bot')
    parser.add_argument('--model', type=str, help='Path to model', default='models/SpaceInvaders-v0.tfmodel')
    parser.add_argument('--visual', action='store_true')
    args = parser.parse_args()

    env = gym.make('SpaceInvaders-v0')  # create the game
    env.seed(0)  # make results reproducible
    rewards = []

    model = a3c_model(load=args.model)
    for _ in range(num_episodes):
        episode_reward = 0
        states = [downsample(env.reset())]
        while True:
            if len(states) < 4:
                action = env.action_space.sample()
            else:
                frames = np.concatenate(states[-4:], axis=3)
                action = np.argmax(model([frames]))
            if args.visual:
                env.render()
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
