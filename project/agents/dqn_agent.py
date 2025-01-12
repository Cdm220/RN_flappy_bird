import torch
import torch.optim as optim
import torch.nn as nn
import random
from models.dqn import DQN
from models.replay_buffer import ReplayBuffer
from config import Config


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = Config.EPSILON_START
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.memory = ReplayBuffer(Config.MEMORY_SIZE)
        self.steps = 0

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < Config.MIN_MEMORY_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(Config.BATCH_SIZE)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + Config.GAMMA * next_q * (1 - dones)
            target_q = target_q.unsqueeze(1)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(Config.EPSILON_END, self.epsilon * Config.EPSILON_DECAY)

        self.steps += 1
        if self.steps % Config.TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())
