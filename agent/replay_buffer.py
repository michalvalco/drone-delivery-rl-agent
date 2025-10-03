
# replay_buffer.py
"""
Experience replay buffer for DQN training.

Author: Michal Valčo
"""

import random
from collections import deque
from typing import List
import numpy as np

# Import Transition from dqn_agent to ensure we use the same class
from .dqn_agent import Transition


class ReplayBuffer:
    """Fixed-size buffer to store experience transitions."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Store as Transition object (imported from dqn_agent)
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of sampled Transition objects
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (optional advanced feature).
    
    Uses sum-tree data structure for efficient sampling based on TD-error priorities.
    Currently a placeholder for future implementation.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, priority: float = 1.0):
        """Add a transition with priority."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample based on priorities (simplified version - uses weighted sampling)."""
        if len(self.buffer) == 0:
            return []
        
        # Normalize priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
