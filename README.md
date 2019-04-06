# [How to Build a Bot for Atari with OpenAI Gym](https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym)

**Want an in-person tutorial with step-by-step walkthroughs and explanations? See the corresponding AirBnb experience for both beginner and experienced coders alike, at ["Build a Dog Filter with Computer Vision"](https://abnb.me/UunXrPyqVO)** ([See the 45+ 5-star reviews](https://www.airbnb.com/users/show/87172280))

This repository includes all source code for the [tutorial on DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym) with the same title, including:
- Q-table based agent for FrozenLake
- Simple neural network q-learning agent for FrozenLake
- Least squares q-learning agent for FrozenLake
- Code to use fully pretrained Deep Q-learning Network (DQN) agent on Space Invaders

> Each of these agents solve FrozenLake in 5000 episodes or fewer; whereas not in record time or even close to it, the agents are written with minimal tuning

created by [Alvin Wan](http://alvinwan.com), January 2018

![agent](https://user-images.githubusercontent.com/2068077/55676351-857eff80-5888-11e9-8fe6-8acf239f3e50.gif)

# Getting Started

For complete step-by-step instructions, see the [tutorial on DigitalOcean](https://www.digitalocean.com/community/tutorials/how-to-build-atari-bot-with-openai-gym). This codebase was developed and tested using `Python 3.6`. If you're familiar with Python, then see the below to skip the tutorial and get started quickly:

> (Optional) [Setup a Python virtual environment](https://www.digitalocean.com/community/tutorials/common-python-tools-using-virtualenv-installing-with-pip-and-managing-packages#a-thorough-virtualenv-how-to) with Python 3.6.

1. Navigate to the repository root, and install all Python dependencies.

```
pip install -r requirements.txt
```

2. Navigate into `src`.

```
cd src
```

3. Download the Tensorflow model for SpaceInvaders, from Tensorpack's A3C-Gym sample.

```
mkdir models
wget http://models.tensorpack.com/OpenAIGym/SpaceInvaders-v0.tfmodel -O models/SpaceInvaders-v0.tfmodel
```

4. Launch the script to see the Space Invaders agent in action.

```
python bot_6_dqn.py --visual
```

# How it Works

See the below resources for explanations of related concepts:

- ["Understanding Deep Q-Learning"](http://alvinwan.com/understanding-deep-q-learning)
- ["Understanding the Bias-Variance Tradeoff"](http://alvinwan.com/understanding-the-bias-variance-tradeoff)

