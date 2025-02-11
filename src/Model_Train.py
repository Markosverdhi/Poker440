import gym
import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# If using Jupyter, enable inline plotting (ignore this line if running as a script)
%matplotlib inline

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
# Attempt to import your custom Texas Hold'em environment.
# Replace 'your_module' with the actual module name where your environment is defined.
try:
    from enviro import TexasHoldem6PlayerEnv
    env = TexasHoldem6PlayerEnv()
except ImportError:
    print("womp womp")
# -----------------------------------------------------------------------------
# Hyperparameters and Device Setup
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
GAMMA = 0.99
LEARNING_RATE = 1e-3
MEMORY_CAPACITY = 10000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 10  # update target network every 10 episodes
NUM_EPISODES = 200       # adjust as needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Replay Memory and Transition
# -----------------------------------------------------------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# -----------------------------------------------------------------------------
# Define the DQN Model
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Helper Functions for Action Selection and Optimization
# -----------------------------------------------------------------------------
def select_action(state, policy_net, epsilon, action_space):
    """
    Returns an action based on epsilon-greedy strategy.
    """
    if random.random() < epsilon:
        return action_space.sample()  # Random action for exploration
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Convert batch data to tensors
    state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
    action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
    next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
    done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
    
    # Current Q values for taken actions
    current_q_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute target Q values using the target network
    next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
    expected_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
    
    # Compute mean squared error loss and perform backpropagation
    loss = F.mse_loss(current_q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train_dqn():
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_CAPACITY)

    epsilon = EPS_START
    all_rewards = []
    
    for episode in range(NUM_EPISODES):
        state = env.reset()
        # For newer gym versions, reset() may return a tuple (state, info)
        if isinstance(state, tuple):
            state = state[0]
        done = False
        total_reward = 0
        
        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, done, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            total_reward += reward
            
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            optimize_model(memory, policy_net, target_net, optimizer)
        
        all_rewards.append(total_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        print(f"Episode {episode+1}/{NUM_EPISODES} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    # Save the trained model and rewards log
    torch.save(policy_net.state_dict(), "dqn_poker_model.pth")
    np.save("rewards.npy", np.array(all_rewards))
    print("Training complete. Model and rewards saved.")
    
    return all_rewards

if __name__ == '__main__':
    rewards = train_dqn()