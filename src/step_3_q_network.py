"""
Step 3 -- Use Q-learning network to train bot
"""

import gym
import numpy as np
import random
import tensorflow as tf
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

num_episodes = 2500
discount_factor = 0.99
learning_rate = 0.1
report_interval = 500
exploration_probability = lambda episode: 50. / (episode + 10)


def one_hot(i, n):
    return np.identity(n)[i: i+1]


def main():
    global exploration_probability
    env = gym.make('FrozenLake-v0')  # create the game
    env.seed(0)  # make results reproducible
    n_obs, n_actions = env.observation_space.n, env.action_space.n
    rewards = []

    # 1. Setup placeholders
    obs_t_ph = tf.placeholder(shape=[1, 16], dtype=tf.float32)
    obs_tp1_ph = tf.placeholder(shape=[1,16], dtype=tf.float32)
    act_ph = tf.placeholder(tf.int32, shape=())
    rew_ph = tf.placeholder(shape=(), dtype=tf.float32)
    Q2_tp1_ph = tf.placeholder(shape=[1, 4], dtype=tf.float32)

    # 2. Setup computation graph
    W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
    q_current = tf.matmul(obs_t_ph, W)
    q_target = tf.matmul(obs_tp1_ph, W)

    q_target_max = tf.reduce_max(Q2_tp1_ph, axis=1)
    q_target_sa = rew_ph + discount_factor * q_target_max
    q_current_sa = q_current[0, act_ph]
    error = tf.reduce_sum(tf.square(q_target_sa - q_current_sa))
    pred_act_ph = tf.argmax(q_current, 1)

    # 3. Setup optimization
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(error)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for episode in range(1, num_episodes + 1):
            obs_t = env.reset()
            episode_reward = 0
            while True:
                obs_t_oh = one_hot(obs_t, n_obs)

                # 4. Take step using best action or random action
                action = session.run(pred_act_ph, feed_dict={obs_t_ph: obs_t_oh})[0]
                if np.random.rand(1) < exploration_probability(episode):
                    action = env.action_space.sample()
                obs_tp1, reward, done, _ = env.step(action)

                # 5. Train model
                obs_tp1_oh, obs_t = one_hot(obs_tp1, n_obs), obs_tp1
                Q_tp1 = session.run(q_target, feed_dict={obs_tp1_ph: obs_tp1_oh})
                session.run(update_model, feed_dict={
                    obs_t_ph: obs_t_oh,
                    rew_ph: reward,
                    Q2_tp1_ph: Q_tp1,
                    act_ph: action
                })
                episode_reward += reward
                if done:
                    rewards.append(episode_reward)
                    if episode % report_interval == 0:
                        print('Average Reward %.2f (Episode %d)' % (
                            np.mean(rewards), episode))
                    break
        print('Average reward: %.2f' % np.mean(rewards))

if __name__ == '__main__':
    # pass
    main()

