"""
Step 2 -- Build simple q-learning agent for FrozenLake
"""

import gym
import numpy as np
import random
random.seed(0)  # make results reproducible
np.random.seed(0)  # make results reproducible

num_episodes = 20000
discount_factor = 0.85
learning_rate = 0.9
report_interval = 500
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f (Episode %d)'


def train(X, y, old_model=None, alpha=0.5):
    model = np.linalg.pinv(X.T.dot(X)).dot(X.T.dot(y))
    if old_model is not None:
        model = alpha * model + (1 - alpha) * old_model
    return model


def initialize(n_obs, n_actions):
    # return np.random.random((n_obs, n_actions))  # incorrect version
    return np.random.normal(0.0, 0.1, (n_obs, n_actions))

def Q(model, X):
    return X.dot(model)


def one_hot(i, n):
    return np.identity(n)[i]


def print_report(rewards, episode):
    print(report % (
        np.mean(rewards[-100:]),
        max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
        np.mean(rewards),
        episode))


def main():
    env = gym.make('FrozenLake-v0')  # create the game
    env.seed(0)  # make results reproducible
    n_obs, n_actions = env.observation_space.n, env.action_space.n
    rewards = []

    M1 = initialize(n_obs, n_actions)
    states, labels = [], []
    for episode in range(1, num_episodes + 1):
        if len(states) >= 10000:
            states, labels = [], []
        state = one_hot(env.reset(), n_obs)
        episode_reward = 0
        while True:
            states.append(state)
            noise = np.random.random((1, n_actions)) / episode
            action = np.argmax(Q(M1, state) + noise)
            state2, reward, done, _ = env.step(action)
            state2 = one_hot(state2, n_obs)
            Qtarget = reward + discount_factor * np.max(Q(M1, state2))

            Qout = Q(M1, state)
            Qlabel = (1 - learning_rate) * Qout[action] + learning_rate * Qtarget
            mask = one_hot(action, n_actions)
            label = (1 - mask) * Qout + mask * Qlabel

            labels.append(label)
            episode_reward += reward
            state = state2
            if len(states) % 10 == 0:
                M1 = train(np.array(states), np.array(labels), M1)
            if done:
                rewards.append(episode_reward)
                if episode % report_interval == 0:
                    print_report(rewards, episode)
                break
    print_report(rewards, -1)

if __name__ == '__main__':
    main()
