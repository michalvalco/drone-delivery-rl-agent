
"""
Entity classes for Drone Delivery Environment
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Package:
    position: np.ndarray
    destination: np.ndarray
    priority: int
    picked_up: bool = False
    delivered: bool = False
    
    def __post_init__(self):
        # Ensure numpy arrays
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.int32)
        if not isinstance(self.destination, np.ndarray):
            self.destination = np.array(self.destination, dtype=np.int32)
        
        # Validate priority
        if self.priority < 1 or self.priority > 3:
            raise ValueError("Priority must be between 1 and 3")


@dataclass
class ChargingStation:
    position: np.ndarray
    charging_rate: float = 0.1
    
    def __post_init__(self):
        # Ensure numpy array
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=np.int32)
        
        # Validate charging rate
        if self.charging_rate <= 0 or self.charging_rate > 1:
            raise ValueError("Charging rate must be between 0 and 1")
