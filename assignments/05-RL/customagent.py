import gymnasium as gym
import torch
from collections import deque, namedtuple
import numpy as np
import random

# %%
# [Algorithm]

# Initialize replay memory D to capacity N.
# Initialize action-value function Q (DQN network 1) with random weights.
# Initialize target action-value function Q (DQN network 2) with weights same to first DQN network.

# For episode 1 to M, do (tracks number of episodes):
# - initialize sequence. For lunar lander, no need for special preprocessing.
# - for t = 1 to T, do (tracks total number of states in episode, until terminal state is reach)
#     - use the epsilon greedy strategy: with probability e select a random action a, otherwise select the ideal action that maximizes Q value according to action-value function (DQN 1)
#     - execute action to observe reward and next state. store this into experience of (state, action, reward, and next state) into the replay memory D
#     - draw a minibatch of the random samples (of state-action-reward-nextstate pairings) from memory D
#     - compute the **Q-value** using the target network by accounting for terminal states.
#     - compute the **predicted Q-values**, using the online network.
#     - perform a gradient descent step between the target and predicted Q-values to update the online network (DQN network 1)
#     - at regular intervals (e.g. every C steps), copy the weights of online network (DQN1) to target network (DQN2)

# %%
# [Environment]

# env = gym.make("LunarLander-v2", render_mode = "none")
# observation, info = env.reset(seed=42)

# action_space_size = env.action_space.n
# observation_space_size = env.observation_space.shape[0]


# %%
# Define the DQN network architecture. Both the online & target QNet.
# The online QNet is the one that is updated at each iteration. The target QNet is the one that is used to calculate the target Q value. The target QNet is updated every C iterations.
# Current state as input, Q-values associated with each state as output
class dqn_net(torch.nn.Module):
    def __init__(self, obs_size, act_size):
        super(dqn_net, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 30)
        self.fc2 = torch.nn.Linear(30, 80)
        self.fc3 = torch.nn.Linear(80, 30)
        self.fc4 = torch.nn.Linear(30, act_size)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


# %%
# Define the named tuple for storing experience replay
Experience = namedtuple(
    "Experience", ("observation", "action", "reward", "next_observation", "done")
)


# %%
# Encapsulates logic about the behavior of the agent
class Agent:
    """
    The DQN agent does all of da learnin
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        Initialize the agent.
        """
        # Store action spaces, observation spaces, and their sizes
        self.action_space_size = (
            action_space.n
        )  # env.action_space: Discrete(4), the valid action choices would be integers 0, 1, 2, and 3
        self.observation_space_size = observation_space.shape[
            0
        ]  # env.observation_space: 8-dimensional vector
        self.action_space = action_space
        self.observation_space = observation_space
        self.epsilon = 1.0

        # Replay Memory. # Replay memory stores all of the interactions within the environment; similar to "dataset" in supervised learning. At each iteration, feed a minibatch of data from the replay memory to the network to make sure the samples are not correlated & to improve sample efficiency. Initialize replay memory to capacity N.
        self.replay_memory = deque(maxlen=500000)  # deque of experiences
        self.previous_observation = (
            None  # will be needed because of this learn function
        )
        self.previous_action = None  # will be needed because of this learn function
        self.number_of_samples = 0

        # Q & Target Network Shananigans
        self.q_net = dqn_net(observation_space.shape[0], action_space.n)
        self.t_net = dqn_net(observation_space.shape[0], action_space.n)
        self.t_net.load_state_dict(self.q_net.state_dict())  # sets it equal to q_net
        self.t_net.eval()  # sets it to inference mode, always(TM)

        # Training for Q-Network
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.01)
        self.loss_fn = torch.nn.MSELoss()

    # how does the agent act? epsilon greedy policy -- if r<epsilon, perform random action; else, perform action with highest Q-value. Start epsilon at *epsilon*, and decay it by *epsilon_decay* every iteration until it reaches *epsilon_min*.
    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        The act method should take an observation and return an action.
        """
        self.previous_observation = observation  # store the previous observation
        if np.random.rand() <= self.epsilon:
            self.previous_action = self.action_space.sample()
            return self.previous_action
        else:
            with torch.no_grad():  # because we don't want the Q-network to be updated
                q_values = self.q_net(
                    torch.tensor(observation)
                )  # only returns q values for each state in the positions 0, 1, 2, 3
                self.previous_action = torch.argmax(q_values).item()
                return self.previous_action

    def replay_memory_sample_minibatch(self):
        """
        Sample a batch of experiences from the replay memory.
        """

    # how does the agent learn? Q values calculate the discounted sum of rewards. since we don't have a ground truth, a recursive formulation (where predicted value gradually matches target value) allows us to compute the target value which is used to update the online network. We use the standard mean squared error loss on the target value and predicted Q-value to update the online network. We use states from the minibatch of experiences --> feed them into online Q-net, pick Q-value associated with the corresponding actions in the minibatch. As for the target, we sum together the reward observed with q-value of next state and pair (using the target q-net). We also need to check if it's the final state of the episode, known as the terminal state -- if so, we're sure of no future rewards and the target value is just the reward.
    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        The learn method should take an observation, a reward, a boolean indicating whether the episode has terminated, and a boolean indicating whether the episode was truncated.
        """
        self.replay_memory.append(
            Experience(
                self.previous_observation,
                self.previous_action,
                reward,
                observation,
                terminated,
            )
        )  # let's look at the previous experiences
        self.number_of_samples += 1

        if (
            len(self.replay_memory) >= 100
        ):  # only if enough experiences to form a minibatch
            # Sample a minibatch of experiences from the replay memory
            random.sample(self.replay_memory, 100)
            minibatch = random.sample(
                self.replay_memory, 100
            )  # also does not sample with replacement
            observations, actions, rewards, next_observations, dones = zip(
                *minibatch
            )  # [(observation_1, observation_2, ...), (action_1, action_2, ...), (reward_1, reward_2, ...), (next_observation_1, next_observation_2, ...), (done_1, done_2, ...)]
            observations = torch.Tensor(observations)  # shape: (100, 8)
            actions = torch.unsqueeze(torch.Tensor(actions), dim=1).to(
                torch.int64
            )  # weird formatting modification needed: a) only int64 is allowed and b) need to convert actions list into a tensor. shape: (100, 1)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            next_observations = torch.Tensor(next_observations)
            dones = torch.Tensor(dones).unsqueeze(1)
            # return observations, actions, rewards, next_observations, dones
            # observations, actions, rewards, next_observations, dones = self.replay_memory_sample_minibatch()

            # Prediction & loss loop
            prediction_qvals = self.q_net(observations).gather(
                1, actions.to(torch.int64)
            )  # Get predictions. weird error where only int64 is allowed. self.q_net(observations) shape: (100, 4), actions shape: (100, 1). gather() returns a tensor with the same shape as the indices tensor, where each value of the output tensor is taken from the input tensor at the corresponding index. shape: (100, 1)
            next_qvals = (
                self.t_net(next_observations).max(1)[0].unsqueeze(1)
            )  # Compute Q-values for the next observations using target network
            target_qvals = rewards + 0.99 * next_qvals * (
                1 - dones
            )  # Compute target / ideal Q-vals. Gamma = 0.99 here.
            loss = self.loss_fn(prediction_qvals, target_qvals)  # Compute loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Updating t-network
            if self.number_of_samples % 4 == 0:  # update every 4 iterations
                self.t_net.load_state_dict(self.q_net.state_dict())

            # Epsilon decay
            if self.epsilon > 0.1:  # the minimum value epsilon can be
                self.epsilon *= 0.995  # decay rate
