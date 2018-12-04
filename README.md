# Q-Learning Bots for Atari Games
Deep Reinforcement Learning Bot for Atari Games Featuring Human-level Gameplay, written in Tensorflow

In this repository, we explore three different q-learning agents:
- Q-table based agent for FrozenLake
- Simple neural network q-learning agent for FrozenLake
- Least squares q-learning agent for FrozenLake
Each of these agents solve FrozenLake in 5000 episodes or fewer; whereas not in record time or even close to it, the agents are functioning and demonstrate the point.

We additionally add bells and whistles for a fully pretrained Deep Q-learning Network (DQN) agent on Space Invaders.

## Getting Started

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
