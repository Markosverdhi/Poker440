import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Environment import TexasHoldem6PlayerEnv

# Hyperparameters and Device Setup
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_CAPACITY = 10000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  
NUM_EPISODES = 200       

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Memory and Transition
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Helper Functions for Action Selection and Optimization
class ModelTrain:
    def __init__(self, model_class, **kwargs):

        self.env = TexasHoldem6PlayerEnv()
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n

        self.model = model_class(self.input_dim, self.output_dim).to(device)

        self.gamma = kwargs.get("gamma", GAMMA)
        self.epsilon = kwargs.get("eps_start", EPS_START)
        self.eps_end = kwargs.get("eps_end", EPS_END)
        self.eps_decay = kwargs.get("eps_decay", EPS_DECAY)
        self.batch_size = kwargs.get("batch_size", BATCH_SIZE)
        self.target_update_freq = kwargs.get("target_update_freq", TARGET_UPDATE_FREQ)
        self.num_episodes = kwargs.get("num_episodes", NUM_EPISODES)
        self.learning_rate = kwargs.get("learning_rate", LEARNING_RATE)

        # Optimizer setup (defaulting to Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayMemory(kwargs.get("memory_capacity", MEMORY_CAPACITY))

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.model(state_tensor)
                return q_values.argmax(dim=1).item()

    def optimize_model(self):
        """Performs optimization on the model using replay memory."""
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)

        # Compute Q values
        current_q_values = self.model(state_batch).gather(1, action_batch)

        # Compute expected Q values
        next_q_values = self.model(next_state_batch).max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute loss and optimize
        loss = F.mse_loss(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        """Trains the model using the specified algorithm."""
        all_rewards = []

        for episode in range(self.num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            done = False
            total_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                total_reward += reward

                self.memory.push(state, action, reward, next_state, done)
                state = next_state

                self.optimize_model()

            all_rewards.append(total_reward)
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

            if episode % self.target_update_freq == 0:
                self.model.load_state_dict(self.model.state_dict())

            print(f"Episode {episode+1}/{self.num_episodes} - Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        print("Training complete.")
        
        return self.model