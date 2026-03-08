import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DeepEmbeddingDQN(nn.Module):
    def __init__(self, board_size=(3, 3), num_actions=4, num_unique_tiles=9, embedding_dim=4):
        super(DeepEmbeddingDQN, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=num_unique_tiles, embedding_dim=embedding_dim)
        
        height, width = board_size
        flat_size = height * width * embedding_dim
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
    def forward(self, state_indices):
        x = self.embedding(state_indices)
        return self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, next_valid_mask):
        self.buffer.append((state, action, reward, next_state, done, next_valid_mask))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_valid_masks = zip(*batch)
        return (
            np.array(states, dtype=np.int64), 
            np.array(actions, dtype=np.int64), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states, dtype=np.int64), 
            np.array(dones, dtype=np.float32),
            np.array(next_valid_masks, dtype=np.int32)
        )
        
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, board_size=(3, 3), num_actions=4, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=50000):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DeepEmbeddingDQN(board_size, num_actions).to(self.device)
        self.target_network = DeepEmbeddingDQN(board_size, num_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # Target network is not updated by gradients
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
    def select_action(self, state, epsilon, valid_actions=None):
        if random.random() < epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.LongTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)[0]
                
                # Soft Masking: Instead of -inf, we subtract a large penalty from invalid
                # action Q-values. This prevents the agent from selecting them while
                # still allowing the network to maintain its mathematical gradients for them.
                if valid_actions is not None:
                    for i in range(self.num_actions):
                        if i not in valid_actions:
                            q_values[i] -= 1000.0
                            
                return q_values.argmax().item()
                
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0 # Not enough samples to train
            
        states, actions, rewards, next_states, dones, next_valid_masks = self.memory.sample(self.batch_size)
        
        states = torch.LongTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.LongTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        next_valid_masks = torch.BoolTensor(next_valid_masks).to(self.device)
        
        # Calculate current Q values
        # q_network(states) -> [batch_size, num_actions]
        # gather -> gets the Q value space of the action historically taken
        current_q_values = self.q_network(states).gather(1, actions).squeeze(1)
        
        # Calculate target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            
            # Mask out invalid actions in the next state with -inf
            # This is critical to prevent the network from evaluating untrained 
            # dummy Q-values of invalid actions as its highest-value future state!
            next_q_values.masked_fill_(~next_valid_masks, -float('inf'))
            
            # Account for terminal states (where all features might technically be invalid masks)
            max_next_q_values = next_q_values.max(1)[0]
            
            # If the episode is done, mathematically 0 out the next state's value 
            # to handle the edge-case where all -inf causes NaN warnings
            max_next_q_values = torch.where(dones > 0, torch.zeros_like(max_next_q_values), max_next_q_values)

            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
            
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Global gradient clipping to stabilize training without destroying gradient direction
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return loss.item()
        
    def update_target_network(self):
        """Sync target network with current Q network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
