# BreakOut-RL
Apply Deep Reinforcement Learning Methods on Atari Games

## Installations
---
Type the following on cmd prompt to install OpenAI Gym Environements:

- %pip install -U gym>=0.21.0
- %pip install -U gym[atari,accept-rom-license]

For Conda:
- conda install -c conda-forge gym-atari

---

## Implementations
---
**Algorithms:**
The following is the simplest DQN model that doesn't train well when there is a huge amount of information need to be stored.

![alt text](https://user-images.githubusercontent.com/106431527/170825760-6fdccc2c-15ba-440a-85c1-f6d45c9bc5c3.png)

The following is a furthur implementations of DQN model called DDQN. This model uses two Q-value estimators to separate action selection from action evaluation and also utilizes two network model and a replay buffer for model training. One neural network is using experience replay to store last episode and determine the next maximum action selection and uses it to train the other neural network for learning. The separation allows each step to use a different function approximator, this results in a overall better approximation of action-value function. However, there are also better way to improve it.

![alt text](https://user-images.githubusercontent.com/106431527/170825681-b96c892a-d1d9-4499-b2f5-9a2ffcf7086e.png)

A dueling neural network architecture takes in further by introducing a advantage value to show its advantages of choosing that action and separates the representation of estimated state-values and advantages for each action into two different streams from a common convolutional feature 3 learning module. The two streams are then recombined using an aggregating layer to produce Q value estimations for each function. The dueling architecture allows the neural network (NN) to learn which states are valuable without having to learn the effect of each action for each state as it is unnecessary to know the value of each action with each time step.

![alt text](https://user-images.githubusercontent.com/106431527/170825699-ccff4940-693e-477e-8ec2-e21298d10231.png)

---

## Action Space
---
- Breakout Environments have the following 4 actions: ['NOOP', 'FIRE', 'RIGHT', 'LEFT'].

you can use the below code to check what your action space and meaning are:
- print("The environment has the following {} actions: {}".format(env.action_space.n, 
                                                                env.unwrapped.get_action_meanings()))
---

## Hyperparameters
---
- Batch size: 64
- Number of Episode: 100000
- Memory Size: 40000
- update size: 1000
-  replay memory max size: 50000
-  replay memory min size: 40000
-  reward clip: (-1,1)

Image Processor:
- crop size: 34 px top, 26 px bottom
- resize: 84x84
- frame stack: 4

Optimizer: optim.ADAM
- learning rate: 0.0003
- gamma: 0.99
- Epsilon Init: 1
- Epsion Final: 0.05
- Epsilon exponential rate: 0.99

---

## Performance:
---
After 3000 iterations:
![alt text](https://user-images.githubusercontent.com/106431527/170826562-0fbdffed-69f1-441b-8a3b-6ac56c1e7881.png)

**Video:**

https://user-images.githubusercontent.com/106431527/170826633-5a9c9b5a-5df9-4d83-ba44-d9f6a82b2036.mp4

---

## Furthur Improvements:
---
- Prioritised replay
- Human start
- Multi-step learning
- Distributional Reinforcement Learning
- Rainbow DQN

---

## Acknowledgements:
---
- Playing Atari with Deep Learning - Deepmind - Available at: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- Official Pytorch DQN tutorial - Available at: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Neural Network and Hyperparameters informed by: https://lzzmm.github.io/2021/11/05/breakout/

Adaptation of model to breakout learned from:
- https://github.com/AdrianHsu/breakout-Deep-Q-Network
- https://keras.io/examples/rl/deep_q_network_breakout/
- https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26

Use of pytorch in this environment informed by:
- https://www.mlq.ai/deep-reinforcement-learning-pytorch-implementation/
- https://github.com/iKintosh/DQN-breakout-Pytorch
- https://github.com/bhctsntrk/OpenAIPong-DQN
- https://github.com/jasonbian97/Deep-Q-Learning-Atari-Pytorch
