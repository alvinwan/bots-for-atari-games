# Deep Q-Learning Bots for Atari Games
Deep Reinforcement Learning Bot for Atari Games, written in Tensorflow

**Interested in building this bot step-by-step, or curious how it works in more detail? See the article on Digital Ocean ["How to Build a Bot for Atari with OpenAI Gym"](https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym)**

In this repository, we explore three different q-learning agents:
- Q-table based agent for FrozenLake
- Simple neural network q-learning agent for FrozenLake
- Least squares q-learning agent for FrozenLake
Each of these agents solve FrozenLake in 5000 episodes or fewer; whereas not in record time or even close to it, the agents are functioning and demonstrate the point.

We additionally add bells and whistles for a fully pretrained Deep Q-learning Network (DQN) agent on Space Invaders.

## How it Works

See the below resources for explanations of related concepts:

- ["Understanding Deep Q-Learning"](http://alvinwan.com/understanding-deep-q-learning)
- ["Understanding the Bias-Variance Tradeoff"](http://alvinwan.com/understanding-the-bias-variance-tradeoff)
- ["How to Build a Bot for Atari with OpenAI Gym"](https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym)

## Installation

If you're familiar with deep learning and would like to use these materials, feel free to follow installation instructions. Otherwise, I would highly recommend checking out the corresponding Digital Ocean article for step-by-step instructions and walkthroughs. 

This codebase was developed and tested using `Python 3.5`. Install all pip dependencies.

```
pip install gym==0.9.5 gym[atari] tensorflow==1.5.0 tensorpack==0.8.0 numpy==1.14.0 opencv-python==3.3.0.10
```

Navigate to the `src` directory.

```python
cd src
```

Download the Tensorflow model for SpaceInvaders, from Tensorpack's A3C-Gym sample.

```
mkdir models
wget http://models.tensorpack.com/OpenAIGym/SpaceInvaders-v0.tfmodel -O models/SpaceInvaders-v0.tfmodel
```

Run the corresponding file.

```
python bot_6_dqn.py --visual
```
