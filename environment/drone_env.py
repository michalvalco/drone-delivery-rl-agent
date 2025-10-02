
"""
Fully Fixed Drone Delivery Environment
Fixes: 
1) Fixed environment across episodes
2) Battery consumption reduced + max_steps increased to 250
3) Package carrying/delivery logic fixed
4) Visual legend added
5) Drone starts at charging station
6) Better reward for charging when low battery
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any
from .entities import Package, ChargingStation


class DroneDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        grid_size: int = 20,
        max_steps: int = 250,  # INCREASED from 200
        num_packages: int = 3,
        num_stations: int = 4,
        render_mode: Optional[str] = None,
        fixed_env: bool = True  # NEW: Fixed environment option
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.num_packages = num_packages
        self.num_stations = num_stations
        self.render_mode = render_mode
        self.fixed_env = fixed_env
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(5)
        
        self.drone_pos = np.array([0, 0], dtype=np.int32)
        self.battery = 1.0
        self.current_step = 0
        self.carrying_package = False
        self.current_package = None  # NEW: Track which package we're carrying
        self.packages = []
        self.charging_stations = []
        self.delivered_count = 0
        
        # NEW: Store initial environment for fixed training
        self.initial_packages = None
        self.initial_stations = None
        
        self.window = None
        self.clock = None
        self.cell_size = 30
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Reset drone state
        self.delivered_count = 0
        self.carrying_package = False
        self.current_package = None
        self.current_step = 0
        self.battery = 1.0
        
        # Generate or restore environment
        if self.fixed_env and self.initial_packages is not None:
            # Use fixed environment
            self.packages = []
            for pkg_data in self.initial_packages:
                pkg = Package(
                    position=np.array(pkg_data['position'], dtype=np.int32),
                    destination=np.array(pkg_data['destination'], dtype=np.int32),
                    priority=pkg_data['priority']
                )
                pkg.picked_up = pkg_data['picked_up']
                pkg.delivered = pkg_data['delivered']
                self.packages.append(pkg)
            
            self.charging_stations = []
            for station_data in self.initial_stations:
                station = ChargingStation(
                    position=np.array(station_data['position'], dtype=np.int32),
                    charging_rate=station_data['charging_rate']
                )
                self.charging_stations.append(station)
        else:
            # Generate new environment
            self.packages = []
            for i in range(self.num_packages):
                priority = self.np_random.integers(1, 4)
                pkg = Package(
                    position=self.np_random.integers(0, self.grid_size, size=2),
                    destination=self.np_random.integers(0, self.grid_size, size=2),
                    priority=priority
                )
                self.packages.append(pkg)
            
            self.charging_stations = []
            for _ in range(self.num_stations):
                station = ChargingStation(
                    position=self.np_random.integers(0, self.grid_size, size=2),
                    charging_rate=0.1
                )
                self.charging_stations.append(station)
            
            # Store initial environment if using fixed mode
            if self.fixed_env:
                self.initial_packages = [
                    {
                        'position': pkg.position.copy(),
                        'destination': pkg.destination.copy(),
                        'priority': pkg.priority,
                        'picked_up': pkg.picked_up,
                        'delivered': pkg.delivered
                    }
                    for pkg in self.packages
                ]
                self.initial_stations = [
                    {
                        'position': station.position.copy(),
                        'charging_rate': station.charging_rate
                    }
                    for station in self.charging_stations
                ]
        
        # NEW: Start drone at a charging station (random one if multiple)
        if self.charging_stations:
            start_station = self.charging_stations[self.np_random.integers(0, len(self.charging_stations))]
            self.drone_pos = np.array(start_station.position.copy())
        else:
            self.drone_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int32)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        terminated = False
        truncated = False
        
        # Movement actions (0-3)
        if action < 4:
            old_pos = self.drone_pos.copy()
            
            if action == 0:  # North
                self.drone_pos[1] = max(0, self.drone_pos[1] - 1)
            elif action == 1:  # South
                self.drone_pos[1] = min(self.grid_size - 1, self.drone_pos[1] + 1)
            elif action == 2:  # East
                self.drone_pos[0] = min(self.grid_size - 1, self.drone_pos[0] + 1)
            elif action == 3:  # West
                self.drone_pos[0] = max(0, self.drone_pos[0] - 1)
            
            # FIXED: Reduced battery consumption
            if not np.array_equal(old_pos, self.drone_pos):
                self.battery -= 0.01  # Changed from 0.02
                reward -= 1
        
        # Context-sensitive action (4): Pickup, Deliver, or Charge
        elif action == 4:
            action_performed = False
            
            # FIXED: Proper delivery check using current_package
            if self.carrying_package and self.current_package is not None:
                if np.array_equal(self.drone_pos, self.current_package.destination):
                    self.current_package.delivered = True
                    self.carrying_package = False
                    # Reward: base + priority bonus
                    reward += 100 + (self.current_package.priority * 50)
                    self.delivered_count += 1
                    self.current_package = None
                    action_performed = True
            
            # FIXED: Proper pickup logic
            if not action_performed and not self.carrying_package:
                for pkg in self.packages:
                    if not pkg.picked_up and np.array_equal(self.drone_pos, pkg.position):
                        pkg.picked_up = True
                        self.carrying_package = True
                        self.current_package = pkg  # Store reference
                        reward += 10
                        action_performed = True
                        break
            
            # Charging logic
            if not action_performed:
                for station in self.charging_stations:
                    if np.array_equal(self.drone_pos, station.position):
                        if self.battery < 1.0:
                            old_battery = self.battery
                            self.battery = min(1.0, self.battery + station.charging_rate)
                            # NEW: Extra reward for charging when battery is low
                            if old_battery < 0.3:
                                reward += 15  # Higher reward for strategic charging
                            else:
                                reward += 5
                            action_performed = True
                        break
            
            # Penalty for unsuccessful context action
            if not action_performed:
                reward -= 5
        
        self.current_step += 1
        
        # Check termination conditions
        if self.battery <= 0:
            terminated = True
            reward -= 50
        
        if self.delivered_count == self.num_packages:
            terminated = True
            reward += 100  # Bonus for completing all deliveries
        
        if self.current_step >= self.max_steps:
            truncated = True
            if self.delivered_count < self.num_packages:
                reward -= 30
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(12, dtype=np.float32)
        
        # Drone position (normalized)
        obs[0] = self.drone_pos[0] / self.grid_size
        obs[1] = self.drone_pos[1] / self.grid_size
        
        # Battery level
        obs[2] = self.battery
        
        # Carrying package indicator
        obs[3] = 1.0 if self.carrying_package else 0.0
        
        # FIXED: Proper current package identification
        current_pkg = None
        if self.carrying_package and self.current_package is not None:
            current_pkg = self.current_package
        else:
            # Find next package to pick up
            for pkg in self.packages:
                if not pkg.picked_up:
                    current_pkg = pkg
                    break
        
        # Package information
        if current_pkg:
            obs[4] = current_pkg.priority / 3.0
            if self.carrying_package:
                # Show destination when carrying
                obs[5] = current_pkg.destination[0] / self.grid_size
                obs[6] = current_pkg.destination[1] / self.grid_size
            else:
                # Show pickup location when not carrying
                obs[5] = current_pkg.position[0] / self.grid_size
                obs[6] = current_pkg.position[1] / self.grid_size
        else:
            obs[4] = 0.0
            obs[5] = 0.0
            obs[6] = 0.0
        
        # Nearest charging station
        if self.charging_stations:
            nearest_station = min(
                self.charging_stations,
                key=lambda s: np.linalg.norm(self.drone_pos - s.position)
            )
            obs[7] = nearest_station.position[0] / self.grid_size
            obs[8] = nearest_station.position[1] / self.grid_size
        else:
            obs[7] = 0.0
            obs[8] = 0.0
        
        # Distance to current target
        if current_pkg:
            target = current_pkg.destination if self.carrying_package else current_pkg.position
            obs[9] = np.linalg.norm(self.drone_pos - target) / (self.grid_size * np.sqrt(2))
        else:
            obs[9] = 0.0
        
        # Distance to nearest charging station
        if self.charging_stations:
            obs[10] = np.linalg.norm(self.drone_pos - nearest_station.position) / (self.grid_size * np.sqrt(2))
        else:
            obs[10] = 0.0
        
        # Time remaining (normalized)
        obs[11] = 1.0 - (self.current_step / self.max_steps)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        return {
            "drone_position": self.drone_pos.tolist(),
            "battery": self.battery,
            "carrying_package": self.carrying_package,
            "delivered_count": self.delivered_count,
            "current_step": self.current_step,
            "packages_remaining": sum(1 for pkg in self.packages if not pkg.picked_up),
            "deliveries": self.delivered_count  # For diagnostics
        }
    
    def render(self):
        if self.render_mode is None:
            return
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            # NEW: Increased window size for legend
            window_width = self.grid_size * self.cell_size + 200  # Extra space for legend
            window_height = self.grid_size * self.cell_size + 100
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Drone Delivery Environment")
            self.clock = pygame.time.Clock()
        
        # NEW: Canvas with room for legend
        canvas_width = self.grid_size * self.cell_size + 200
        canvas_height = self.grid_size * self.cell_size + 100
        canvas = pygame.Surface((canvas_width, canvas_height))
        canvas.fill((255, 255, 255))
        
        # Draw grid
        for x in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (x * self.cell_size, 0),
                (x * self.cell_size, self.grid_size * self.cell_size)
            )
        for y in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                (200, 200, 200),
                (0, y * self.cell_size),
                (self.grid_size * self.cell_size, y * self.cell_size)
            )
        
        # Draw charging stations
        for station in self.charging_stations:
            pos = (station.position[0] * self.cell_size, station.position[1] * self.cell_size)
            pygame.draw.rect(
                canvas,
                (255, 200, 0),  # Yellow
                (pos[0] + 5, pos[1] + 5, self.cell_size - 10, self.cell_size - 10)
            )
        
        # Draw packages and destinations
        for pkg in self.packages:
            # Draw package (if not picked up)
            if not pkg.picked_up:
                pos = (pkg.position[0] * self.cell_size, pkg.position[1] * self.cell_size)
                color = (
                    (0, 255, 0) if pkg.priority == 1 else 
                    (255, 165, 0) if pkg.priority == 2 else 
                    (255, 0, 0)
                )
                pygame.draw.circle(
                    canvas,
                    color,
                    (pos[0] + self.cell_size // 2, pos[1] + self.cell_size // 2),
                    self.cell_size // 3
                )
            
            # Draw destination (if not delivered)
            if not pkg.delivered:
                dest_pos = (pkg.destination[0] * self.cell_size, pkg.destination[1] * self.cell_size)
                color = (
                    (150, 255, 150) if pkg.priority == 1 else 
                    (255, 215, 150) if pkg.priority == 2 else 
                    (255, 150, 150)
                )
                pygame.draw.rect(
                    canvas,
                    color,
                    (dest_pos[0] + 8, dest_pos[1] + 8, self.cell_size - 16, self.cell_size - 16)
                )
        
        # Draw drone - FIXED: Different color when carrying
        drone_pos = (self.drone_pos[0] * self.cell_size, self.drone_pos[1] * self.cell_size)
        drone_color = (128, 0, 255) if self.carrying_package else (0, 0, 255)  # Purple when carrying
        pygame.draw.circle(
            canvas,
            drone_color,
            (drone_pos[0] + self.cell_size // 2, drone_pos[1] + self.cell_size // 2),
            self.cell_size // 2 - 3
        )
        
        # Draw info panel at bottom
        info_y = self.grid_size * self.cell_size + 10
        font = pygame.font.Font(None, 24)
        
        battery_text = font.render(f"Battery: {self.battery:.2f}", True, (0, 0, 0))
        canvas.blit(battery_text, (10, info_y))
        
        delivered_text = font.render(f"Delivered: {self.delivered_count}/{self.num_packages}", True, (0, 0, 0))
        canvas.blit(delivered_text, (200, info_y))
        
        step_text = font.render(f"Step: {self.current_step}/{self.max_steps}", True, (0, 0, 0))
        canvas.blit(step_text, (400, info_y))
        
        carrying_text = font.render(f"Carrying: {'Yes' if self.carrying_package else 'No'}", True, (0, 0, 0))
        canvas.blit(carrying_text, (10, info_y + 30))
        
        # NEW: Draw legend on the right side
        legend_x = self.grid_size * self.cell_size + 10
        legend_y = 10
        legend_font = pygame.font.Font(None, 20)
        
        # Legend title
        title = legend_font.render("LEGEND:", True, (0, 0, 0))
        canvas.blit(title, (legend_x, legend_y))
        legend_y += 25
        
        # Charging station
        pygame.draw.rect(canvas, (255, 200, 0), (legend_x, legend_y, 15, 15))
        text = legend_font.render("Charging Station", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Package Priority 1
        pygame.draw.circle(canvas, (0, 255, 0), (legend_x + 7, legend_y + 7), 7)
        text = legend_font.render("Package (P1)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Package Priority 2
        pygame.draw.circle(canvas, (255, 165, 0), (legend_x + 7, legend_y + 7), 7)
        text = legend_font.render("Package (P2)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Package Priority 3
        pygame.draw.circle(canvas, (255, 0, 0), (legend_x + 7, legend_y + 7), 7)
        text = legend_font.render("Package (P3)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Destination Priority 1
        pygame.draw.rect(canvas, (150, 255, 150), (legend_x, legend_y, 15, 15))
        text = legend_font.render("Destination (P1)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Destination Priority 2
        pygame.draw.rect(canvas, (255, 215, 150), (legend_x, legend_y, 15, 15))
        text = legend_font.render("Destination (P2)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Destination Priority 3
        pygame.draw.rect(canvas, (255, 150, 150), (legend_x, legend_y, 15, 15))
        text = legend_font.render("Destination (P3)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Drone (empty)
        pygame.draw.circle(canvas, (0, 0, 255), (legend_x + 7, legend_y + 7), 7)
        text = legend_font.render("Drone (empty)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        legend_y += 25
        
        # Drone (carrying)
        pygame.draw.circle(canvas, (128, 0, 255), (legend_x + 7, legend_y + 7), 7)
        text = legend_font.render("Drone (carrying)", True, (0, 0, 0))
        canvas.blit(text, (legend_x + 20, legend_y))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
