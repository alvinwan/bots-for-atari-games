"""Minimal script for featurizing any input using a3c.

Allows you to pick layer to output from. Graph code below from train-atari.py in A3C-Gym from Tensorpack examples.
"""

import numpy as np
import tensorflow as tf

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.serialize import *
from tensorpack.utils.stats import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient

from tensorpack.RL import PreventStuckPlayer
from tensorpack.RL import GymEnv
from tensorpack.RL import MapPlayerState
from tensorpack.RL import HistoryFramePlayer
from tensorpack.RL import LimitLengthPlayer

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)


LAYERS = ('policy', 'conv0/output', 'pool0/output', 'conv1/output',
          'pool1/output', 'conv2/output', 'pool2/output', 'conv3/output',
          'fc0/output', 'prelu/output', 'fc-pi/output', 'fc-v/output')

__all__ = ['a3c', 'a3c_model']


def a3c_model(layer='policy', load='SpaceInvaders-v0.tfmodel', num_actions=6):
    """Provide a featurizer

    :param layer: the a3c neural network layer to output
    :param load: path to the model to load
    :param num_actions: Number of actions available in the game
    """
    assert layer in LAYERS, 'Invalid layer %s. One of: %s' % (layer, LAYERS)
    cfg = PredictConfig(
        model=Model(num_actions=num_actions),
        session_init=get_model_loader(load),
        input_names=['state'],
        output_names=[layer])
    predfunc = OfflinePredictor(cfg)
    return predfunc


def a3c(state, layer='policy', load='SpaceInvaders-v0.tfmodel', num_actions=6):
    """Featurize the provided state.

    :param state: 84x84x(3 * FRAME_HISTORY_LEN)
    :param layer: the a3c neural network layer to output
    :param load: path to the model to load
    :param num_actions: Number of actions available in the game
    """
    predfunc = a3c_model(layer, load, num_actions)
    return predfunc([[state]])


class Model(ModelDesc):
    """Build the A3C model and post-processing"""

    def __init__(self, num_actions):
        self.num_actions = num_actions

    def _get_inputs(self):
        return [InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward'),
                InputDesc(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, out_dim=self.num_actions, nl=tf.identity)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return logits, value

    def _build_graph(self, inputs):
        state, action, futurereward, action_prob = inputs
        logits, self.value = self._get_NN_prediction(state)
        self.value = tf.squeeze(self.value, [1], name='pred_value')  # (B,)
        self.policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(self.policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, self.num_actions), 1)
        advantage = tf.subtract(tf.stop_gradient(self.value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(self.policy * tf.one_hot(action, self.num_actions), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(
            self.policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(self.value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   self.cost, tf.reduce_mean(importance, name='importance'))

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


if __name__ == '__main__':
    frames = np.random.random((84, 84, 12))
    a3c(frames)
