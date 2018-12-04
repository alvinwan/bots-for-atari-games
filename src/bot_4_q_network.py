"""
Bot 4 -- Use Q-learning network to train bot
"""

from typing import List
import gym
import numpy as np
import random
import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

num_episodes = 4000
discount_factor = 0.99
learning_rate = 0.15
report_interval = 500
exploration_probability = lambda episode: 50. / (episode + 10)
report = '100-ep Average: %.2f . Best 100-ep Average: %.2f . Average: %.2f ' \
         '(Episode %d)'


def one_hot(i: int, n: int) -> np.array:
    """Implements one-hot encoding by selecting the ith standard basis vector"""
    return np.identity(n)[i].reshape((1, -1))


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

    # 1. Setup placeholders
    n_obs, n_actions = env.observation_space.n, env.action_space.n
    obs_t_ph = tf.placeholder(shape=[1, n_obs], dtype=tf.float32)
    obs_tp1_ph = tf.placeholder(shape=[1, n_obs], dtype=tf.float32)
    act_ph = tf.placeholder(tf.int32, shape=())
    rew_ph = tf.placeholder(shape=(), dtype=tf.float32)
    q_target_ph = tf.placeholder(shape=[1, n_actions], dtype=tf.float32)

    # 2. Setup computation graph
    W = tf.Variable(tf.random_uniform([n_obs, n_actions], 0, 0.01))
    q_current = tf.matmul(obs_t_ph, W)
    q_target = tf.matmul(obs_tp1_ph, W)

    q_target_max = tf.reduce_max(q_target_ph, axis=1)
    q_target_sa = rew_ph + discount_factor * q_target_max
    q_current_sa = q_current[0, act_ph]
    error = tf.reduce_sum(tf.square(q_target_sa - q_current_sa))
    pred_act_ph = tf.argmax(q_current, 1)

    # 3. Setup optimization
    trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    update_model = trainer.minimize(error)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for episode in range(1, num_episodes + 1):
            obs_t = env.reset()
            episode_reward = 0
            while True:

                # 4. Take step using best action or random action
                obs_t_oh = one_hot(obs_t, n_obs)
                action = session.run(pred_act_ph, feed_dict={obs_t_ph: obs_t_oh})[0]
                if np.random.rand(1) < exploration_probability(episode):
                    action = env.action_space.sample()
                obs_tp1, reward, done, _ = env.step(action)

                # 5. Train model
                obs_tp1_oh = one_hot(obs_tp1, n_obs)
                q_target_val = session.run(q_target, feed_dict={obs_tp1_ph: obs_tp1_oh})
                session.run(update_model, feed_dict={
                    obs_t_ph: obs_t_oh,
                    rew_ph: reward,
                    q_target_ph: q_target_val,
                    act_ph: action
                })
                episode_reward += reward
                obs_t = obs_tp1

                if done:
                    rewards.append(episode_reward)
                    if episode % report_interval == 0:
                        print_report(rewards, episode)
                    break
        print_report(rewards, -1)

if __name__ == '__main__':
    main()
