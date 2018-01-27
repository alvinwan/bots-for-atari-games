"""
Step 2 -- Build simple q-learning agent for FrozenLake
"""

import gym
import numpy as np
import random
random.seed(0)  # make results reproducible
np.random.seed(0)  # make results reproducible

num_episodes = 5000
discount_factor = 0.85
learning_rate = 0.9
report_interval = 500

def main():
    env = gym.make('FrozenLake-v0')  # create the game
    env.seed(0)  # make results reproducible
    rewards = []

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        while True:
            noise = np.random.random((1, env.action_space.n)) / (episode**2.)
            action = np.argmax(Q[state, :] + noise)
            state2, reward, done, _ = env.step(action)
            Qtarget = reward + discount_factor * np.max(Q[state2, :])
            Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * Qtarget
            episode_reward += reward
            state = state2
            if done:
                rewards.append(episode_reward)
                if episode % report_interval == 0:
                    print('Average Reward %.2f (Episode %d)' % (np.mean(rewards), episode))
                break
    print('Average reward: %.2f' % (sum(rewards) / len(rewards)))

if __name__ == '__main__':
    main()
