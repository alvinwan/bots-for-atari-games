"""
Step 1 -- Make a random, baseline agent for the SpaceInvaders game.
"""

import gym
import random


def main():
    env = gym.make('SpaceInvaders-v0')
    num_episodes = 5
    num_actions = env.action_space.n

    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        while True:
            action = random.randint(0, num_actions-1)
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print('Episode %d reward: %d' % (episode, total_reward))
                break


if __name__ == '__main__':
    main()
