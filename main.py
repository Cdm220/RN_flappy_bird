import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import flappy_bird_gymnasium
import gymnasium as gym

GAMMA = 0.95
LEARNING_RATE = 0.0001
BATCH_SIZE = 64 
MEMORY_SIZE = 100000 
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9995 
TARGET_UPDATE_FREQ = 100 
MIN_MEMORY_SIZE = 1000 

def normalize_state(state):
    state = np.array(state, dtype=np.float32)
    state = np.clip(state, -10, 10)
    state = state / 10.0
    return state

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32),
                np.array(next_states), 
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps = 0
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def train(self):
        if len(self.memory) < MIN_MEMORY_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
            target_q = target_q.unsqueeze(1)
        
        loss = nn.SmoothL1Loss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())

def train_agent():
    env = gym.make("FlappyBird-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    best_reward = float('-inf')
    episodes = 5000
    
    for e in range(episodes):
        state, _ = env.reset()
        state = normalize_state(state)
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_state)
            agent.memory.push(state, action, reward, next_state, terminated)
            
            state = next_state
            if reward > 0.1:
                reward = 10.0
            total_reward += reward
            steps += 1
            
            agent.train()
            
            if terminated or truncated:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                'model_state_dict': agent.model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'epsilon': agent.epsilon,
                'best_reward': best_reward
            }, "flappy_bird_dqn3.pth")
        
        print(f"Episode {e + 1}/{episodes}, Steps: {steps}, "
              f"Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}, "
              f"Best Reward: {best_reward:.2f}")
    
    env.close()

def test_agent():
    env = gym.make("FlappyBird-v0", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    checkpoint = torch.load("flappy_bird_dqn3.pth")
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.model.eval()
    
    state, _ = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        state = normalize_state(state)
        action = agent.act(state, training=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        state = next_state
        if reward > 0.1:
            reward = 10.0
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"Test Results - Steps: {steps}, Total Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    # train_agent()
    test_agent()