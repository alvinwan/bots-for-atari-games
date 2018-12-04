"""
Bot 5 -- Build least squares q-learning agent for FrozenLake
"""

from typing import Tuple
from typing import Callable
from typing import List
import gym
import numpy as np
import random
random.seed(0)  # make results reproducible
np.random.seed(0)  # make results reproducible

num_episodes = 5000
discount_factor = 0.85
learning_rate = 0.9
w_lr = 0.5
report_interval = 500
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f ' \
         '(Episode %d)'


def makeQ(model: np.array) -> Callable[[np.array], np.array]:
    """Returns a Q-function, which takes state -> distribution over actions"""
    return lambda X: X.dot(model)


def initialize(shape: Tuple):
    """Initialize model"""
    W = np.random.normal(0.0, 0.1, shape)
    Q = makeQ(W)
    return W, Q


def train(X: np.array, y: np.array, W: np.array) -> Tuple[np.array, Callable]:
    """Train the model, using solution to ridge regression"""
    I = np.eye(X.shape[1])
    newW = np.linalg.inv(X.T.dot(X) + 10e-4 * I).dot(X.T.dot(y))
    W = w_lr * newW + (1 - w_lr) * W
    Q = makeQ(W)
    return W, Q


def one_hot(i: int, n: int) -> np.array:
    """Implements one-hot encoding by selecting the ith standard basis vector"""
    return np.identity(n)[i]


def print_report(rewards: List, episode: int):
    """Print rewards report for current episode
    - Average for last 100 episodes
    - Best 100-episode average across all time
    - Average for all episodes across time
    """
    print(report % (
        np.mean(rewards[-100:]),
        max([np.mean(rewards[i:i+100]) for i in range(len(rewards) - 100)]),
        np.mean(rewards),
        episode))


def main():
    env = gym.make('FrozenLake-v0')  # create the game
    env.seed(0)  # make results reproducible
    rewards = []

    n_obs, n_actions = env.observation_space.n, env.action_space.n
    W, Q = initialize((n_obs, n_actions))
    states, labels = [], []
    for episode in range(1, num_episodes + 1):
        if len(states) >= 10000:
            states, labels = [], []
        state = one_hot(env.reset(), n_obs)
        episode_reward = 0
        while True:
            states.append(state)
            noise = np.random.random((1, n_actions)) / episode
            action = np.argmax(Q(state) + noise)
            state2, reward, done, _ = env.step(action)

            state2 = one_hot(state2, n_obs)
            Qtarget = reward + discount_factor * np.max(Q(state2))
            label = Q(state)
            label[action] = (1 - learning_rate) * label[action] + learning_rate * Qtarget
            labels.append(label)

            episode_reward += reward
            state = state2
            if len(states) % 10 == 0:
                W, Q = train(np.array(states), np.array(labels), W)
            if done:
                rewards.append(episode_reward)
                if episode % report_interval == 0:
                    print_report(rewards, episode)
                break
    print_report(rewards, -1)

if __name__ == '__main__':
    main()
