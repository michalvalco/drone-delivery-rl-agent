
# drone_env.py
"""
DroneDeliveryEnv — Gymnasium-compatible environment for a drone that:
- Picks up highest-priority packages from a warehouse,
- Delivers them to destinations,
- Manages battery and uses chargers strategically.

Key parameters
-------------
fixed_env: bool
    If True, the map (warehouse, chargers, package destinations) is generated once
    and cached; all episodes reuse the same layout for stable learning.
base_seed: int
    Seed used to generate (and cache) the fixed layout.
easy_mode_for_tests: bool
    Diagnostics-only helper. When True:
      * pickup is allowed anywhere inside the warehouse zone (no exact-tile requirement)
      * auto-delivery happens upon stepping on the correct destination (no interact needed)
      * at least one package gets a "short-hop" destination near the warehouse
    Leave False for training/evaluation to keep the intended game rules.

Rewards (final)
---------------
Movement:                 -1    per move
Pickup:                   +10   when successfully picking up a package
Delivery:                 +100 * priority   (priority in {1,2,3})
Charge (battery >= 30%):  +5
Charge (battery < 30%):   +15
Battery death:            -50   (episode ends)
Max steps per episode:     250   (truncate)

Rendering
---------
Pygame grid with a legend panel:
- Yellow squares: chargers
- Gray area: warehouse zone (0..3, 0..3)
- Green/Orange/Red circles: packages by priority (dest squares tinted)
- Blue drone: not carrying; Purple drone: carrying

Author: Michal Valčo
Course: AI Agents – Lesson 10
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Package:
    id: int
    priority: int                 # 1, 2, or 3
    pickup_pos: Tuple[int, int]   # in warehouse
    dest_pos: Tuple[int, int]
    picked_up: bool = False
    delivered: bool = False


# ----------------------------
# Environment
# ----------------------------

class DroneDeliveryEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        grid_size: Tuple[int, int] = (12, 12),
        fixed_env: bool = True,
        base_seed: int = 42,
        n_packages: int = 3,
        n_chargers: int = 4,
        movement_cost: float = 0.01,
        max_steps: int = 250,
        easy_mode_for_tests: bool = False,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.grid_w, self.grid_h = grid_size
        self.fixed_env = fixed_env
        self.base_seed = int(base_seed)
        self.n_packages = int(n_packages)
        self.n_chargers = int(n_chargers)
        self.movement_cost = float(movement_cost)
        self.episode_step_limit = int(max_steps)
        self.easy_mode_for_tests = bool(easy_mode_for_tests)

        # Warehouse zone: a 4x4 block at top-left
        self.wh_x0, self.wh_y0, self.wh_w, self.wh_h = 0, 0, 4, 4
        self.wh_center = (2, 2)

        # Gym API: Discrete actions: 0 up,1 right,2 down,3 left,4 interact
        self.action_space = spaces.Discrete(5)

        # Observation: 12-dim, normalized where applicable
        # [0:1] drone_x_norm, drone_y_norm
        # [2]   battery (0..1)
        # [3:4] dx_to_target_norm, dy_to_target_norm
        # [5]   dist_to_target_norm
        # [6:7] dx_to_charger_norm, dy_to_charger_norm  (nearest charger)
        # [8]   dist_to_charger_norm
        # [9]   carrying (0/1)
        # [10]  time_left_norm (1 - steps/episode_limit)
        # [11]  packages_left_norm
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        # RNG and cached layout
        self._cached_layout = None
        self._np_rng = np.random.default_rng(self.base_seed)

        # Runtime state
        self.packages: List[Package] = []
        self.chargers: List[Tuple[int, int]] = []
        self.drone_pos = np.array([self.wh_center[0], self.wh_center[1]], dtype=np.int32)
        self.battery = 1.0
        self.carrying_package: bool = False
        self.current_package: Optional[Package] = None
        self.steps = 0
        self.screen = None
        self.clock = None
        self.cell_px = 40  # render scale

        self.render_mode = render_mode

    # ----------------------------
    # Gym API
    # ----------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Gymnasium: ensure seeding is wired correctly
        super().reset(seed=seed)

        if self.fixed_env:
            if self._cached_layout is None:
                self._cached_layout = self._generate_layout(self.base_seed)
            layout = self._cached_layout
        else:
            layout = self._generate_layout(seed if seed is not None else self._np_rng.integers(0, 1_000_000))

        self._apply_layout(layout)
        self._init_episode_state()

        if self.render_mode == "human":
            self._ensure_display()

        obs = self._get_obs()
        info = {"seed": seed if seed is not None else self.base_seed}
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)

        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, int | float | str] = {"delivered_now": 0, "charged_now": 0}

        # Move
        old_pos = self.drone_pos.copy()
        if action in (0, 1, 2, 3):
            self._move(action)
            reward += -1.0  # movement penalty
            # battery drains only when actually moved
            if not np.array_equal(self.drone_pos, old_pos):
                self.battery = max(0.0, self.battery - self.movement_cost)

        # Interact: pick up / deliver / charge (order of checks matters)
        elif action == 4:
            action_performed = False  # FIX: Track if interact action was consumed
            
            # Deliver (strict mode requires interact on destination)
            if self.carrying_package and self._at_destination(self.current_package):
                delivery_reward = self._deliver_current(info)
                reward += delivery_reward
                action_performed = True
            
            # Pickup (must be in warehouse; easy mode: anywhere inside zone)
            elif (not self.carrying_package) and self._in_warehouse(self.drone_pos):
                self._pickup_if_available()
                if self.carrying_package:
                    reward += 10.0  # pickup bonus
                    action_performed = True
            
            # Charge (only if didn't pickup or deliver)
            if not action_performed and self._on_charger(self.drone_pos):
                if self.battery < 1.0:  # FIX: Only charge if not full
                    inc = self._charge_amount()
                    before = self.battery
                    self.battery = min(1.0, self.battery + inc)
                    info["charged_now"] = 1
                    reward += 15.0 if before < 0.30 else 5.0
                    action_performed = True
            
            # Invalid interact (nothing to interact with)
            if not action_performed:
                reward += -5.0  # useless interact

        # Easy mode: auto delivery on stepping onto destination (no interact)
        if self.easy_mode_for_tests and self.carrying_package and self._at_destination(self.current_package):
            delivery_reward = self._deliver_current(info)
            reward += delivery_reward

        # Battery death
        if self.battery <= 0.0:
            reward += -50.0
            terminated = True

        self.steps += 1
        if self.steps >= self.episode_step_limit:
            truncated = True

        # All delivered?
        if all(p.delivered for p in self.packages):
            terminated = True

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def render(self):
        self._ensure_display()
        w_px = self.grid_w * self.cell_px + 220  # extra panel for legend
        h_px = self.grid_h * self.cell_px
        surface = pygame.display.get_surface()
        surface.fill((245, 245, 245))

        # Grid
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                r = pygame.Rect(x * self.cell_px, y * self.cell_px, self.cell_px - 1, self.cell_px - 1)
                pygame.draw.rect(surface, (230, 230, 230), r)

        # Warehouse region
        wx0, wy0 = self.wh_x0, self.wh_y0
        r_wh = pygame.Rect(wx0 * self.cell_px, wy0 * self.cell_px,
                           self.wh_w * self.cell_px, self.wh_h * self.cell_px)
        pygame.draw.rect(surface, (210, 210, 210), r_wh)

        # Chargers
        for (cx, cy) in self.chargers:
            rr = pygame.Rect(cx * self.cell_px + 6, cy * self.cell_px + 6, self.cell_px - 12, self.cell_px - 12)
            pygame.draw.rect(surface, (255, 210, 0), rr)

        # Package destinations (light tint by priority)
        for p in self.packages:
            if not p.delivered:
                color = {1: (190, 255, 190), 2: (255, 232, 190), 3: (255, 190, 190)}[p.priority]
                rr = pygame.Rect(p.dest_pos[0] * self.cell_px + 6, p.dest_pos[1] * self.cell_px + 6,
                                 self.cell_px - 12, self.cell_px - 12)
                pygame.draw.rect(surface, color, rr)

        # Packages at warehouse (draw as circles on their pickup_pos)
        for p in self.packages:
            if not p.picked_up:
                px = p.pickup_pos[0] * self.cell_px + self.cell_px // 2
                py = p.pickup_pos[1] * self.cell_px + self.cell_px // 2
                color = {1: (0, 180, 0), 2: (230, 140, 0), 3: (220, 0, 0)}[p.priority]
                pygame.draw.circle(surface, color, (px, py), self.cell_px // 3)

        # Drone
        dcolor = (60, 120, 255) if not self.carrying_package else (150, 80, 255)
        drect = pygame.Rect(self.drone_pos[0] * self.cell_px + 4, self.drone_pos[1] * self.cell_px + 4,
                            self.cell_px - 8, self.cell_px - 8)
        pygame.draw.rect(surface, dcolor, drect)

        # Legend panel
        panel_x = self.grid_w * self.cell_px + 10
        self._draw_legend(surface, panel_x)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    # ----------------------------
    # Helpers
    # ----------------------------

    def _ensure_display(self):
        if self.screen is None:
            pygame.init()
            size = (self.grid_w * self.cell_px + 220, self.grid_h * self.cell_px)
            self.screen = pygame.display.set_mode(size)
            pygame.display.set_caption("Drone Delivery RL")
            self.clock = pygame.time.Clock()

    def _draw_legend(self, surface, x0: int):
        y = 10
        font = pygame.font.SysFont(None, 22)

        def text(line):
            nonlocal y
            surf = font.render(line, True, (30, 30, 30))
            surface.blit(surf, (x0, y))
            y += 22

        text("Legend")
        y += 6
        text("Yellow squares: Charger")
        text("Gray area: Warehouse")
        text("Circles: Packages (G/O/R = 1/2/3)")
        text("Light squares: Destinations")
        text("Blue drone: empty")
        text("Purple drone: carrying")
        y += 10
        text(f"Battery: {self.battery:.2f}")
        text(f"Steps: {self.steps}/{self.episode_step_limit}")
        remaining = sum(1 for p in self.packages if not p.delivered)
        text(f"Packages left: {remaining}")

    def _generate_layout(self, seed: int) -> dict:
        rng = np.random.default_rng(seed)

        # Chargers — ensure at least one inside warehouse
        chargers = set()
        # one inside warehouse
        chargers.add((self.wh_x0 + 1, self.wh_y0 + 1))
        # others across the map
        while len(chargers) < self.n_chargers:
            x = int(rng.integers(0, self.grid_w))
            y = int(rng.integers(0, self.grid_h))
            chargers.add((x, y))
        chargers = list(sorted(chargers))

        # Packages (spawn in warehouse on random cells)
        wh_cells = [(x, y) for x in range(self.wh_w) for y in range(self.wh_h)]
        rng.shuffle(wh_cells)
        packages: List[Package] = []
        for i in range(self.n_packages):
            pickup = (wh_cells[i % len(wh_cells)][0], wh_cells[i % len(wh_cells)][1])
            priority = int(rng.choice([1, 2, 3]))
            dest = self._sample_destination(rng)

            packages.append(Package(
                id=i,
                priority=priority,
                pickup_pos=pickup,
                dest_pos=dest
            ))

        # Easy mode: guarantee one "short-hop" package (dest near warehouse)
        if self.easy_mode_for_tests and packages:
            near = self._nearest_cells_to((self.wh_center[0], self.wh_center[1]), max_radius=3)
            # pick a nearby dest not in warehouse cells
            for cell in near:
                if not self._in_warehouse(np.array(cell)) and 0 <= cell[0] < self.grid_w and 0 <= cell[1] < self.grid_h:
                    packages[0].dest_pos = cell
                    break

        return {"chargers": chargers, "packages": packages}

    def _apply_layout(self, layout: dict):
        self.chargers = list(layout["chargers"])
        self.packages = [Package(**{**p.__dict__}) if isinstance(p, Package) else Package(**p) for p in layout["packages"]]

    def _init_episode_state(self):
        self.drone_pos[:] = np.array(self.wh_center, dtype=np.int32)
        self.battery = 1.0
        self.carrying_package = False
        self.current_package = None
        for p in self.packages:
            p.picked_up = False
            p.delivered = False
        self.steps = 0

    def _move(self, action: int):
        dx, dy = 0, 0
        if action == 0:   # up
            dy = -1
        elif action == 1: # right
            dx = +1
        elif action == 2: # down
            dy = +1
        elif action == 3: # left
            dx = -1
        nx, ny = self.drone_pos[0] + dx, self.drone_pos[1] + dy
        if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
            self.drone_pos[:] = (nx, ny)

    def _in_warehouse(self, pos: np.ndarray | Tuple[int, int]) -> bool:
        x, y = (int(pos[0]), int(pos[1])) if isinstance(pos, np.ndarray) else (int(pos[0]), int(pos[1]))
        return (self.wh_x0 <= x < self.wh_x0 + self.wh_w) and (self.wh_y0 <= y < self.wh_y0 + self.wh_h)

    def _on_charger(self, pos: np.ndarray | Tuple[int, int]) -> bool:
        x, y = (int(pos[0]), int(pos[1])) if isinstance(pos, np.ndarray) else (int(pos[0]), int(pos[1]))
        return (x, y) in self.chargers

    def _at_destination(self, pkg: Optional[Package]) -> bool:
        if pkg is None:
            return False
        return (int(self.drone_pos[0]), int(self.drone_pos[1])) == tuple(pkg.dest_pos)

    def _pickup_if_available(self):
        # Must be standing exactly on a package tile in strict mode
        # In easy mode: anywhere inside warehouse zone
        candidates = [p for p in self.packages if (not p.picked_up) and (not p.delivered)]
        if not candidates:
            return
        # choose highest priority (then nearest destination as tie-breaker)
        candidates.sort(key=lambda p: (-p.priority, self._manhattan(self.drone_pos, p.dest_pos)))

        top = candidates[0]
        if self.easy_mode_for_tests:
            # pickup anywhere inside warehouse
            top.picked_up = True
            self.carrying_package = True
            self.current_package = top
            return

        # strict: must be on the package's pickup_pos
        if tuple(self.drone_pos) == tuple(top.pickup_pos):
            top.picked_up = True
            self.carrying_package = True
            self.current_package = top

    def _deliver_current(self, info: Dict[str, int | float | str]) -> float:
        """Deliver the current package and return the earned reward.
        
        FIX: Changed to return reward instead of storing in broken _pending_reward.
        This was the root cause of zero delivery rewards in diagnostics.
        """
        if not self.carrying_package or self.current_package is None:
            return 0.0
        if not self._at_destination(self.current_package):
            return 0.0
        
        # FIX: Calculate delivery reward (100 × priority)
        delivery_reward = 100.0 * float(self.current_package.priority)
        
        # Update state
        self.current_package.delivered = True
        self.current_package.picked_up = True
        self.carrying_package = False
        self.current_package = None
        info["delivered_now"] = 1
        
        # FIX: Return the reward so step() can add it
        return delivery_reward

    def _charge_amount(self) -> float:
        # Simple fixed increment
        return 0.10

    def _sample_destination(self, rng) -> Tuple[int, int]:
        # Avoid warehouse zone to force navigation
        while True:
            x = int(rng.integers(0, self.grid_w))
            y = int(rng.integers(0, self.grid_h))
            if not self._in_warehouse((x, y)):
                return (x, y)

    def _nearest_cells_to(self, pos: Tuple[int, int], max_radius: int = 3) -> List[Tuple[int, int]]:
        cx, cy = pos
        cells = []
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) + abs(dy) == r:
                        cells.append((cx + dx, cy + dy))
        return cells

    def _manhattan(self, a: Tuple[int, int] | np.ndarray, b: Tuple[int, int] | np.ndarray) -> int:
        ax, ay = (int(a[0]), int(a[1])) if isinstance(a, np.ndarray) else (int(a[0]), int(a[1]))
        bx, by = (int(b[0]), int(b[1])) if isinstance(b, np.ndarray) else (int(b[0]), int(b[1]))
        return abs(ax - bx) + abs(ay - by)

    def _get_obs(self) -> np.ndarray:
        # Build observation
        gx, gy = float(self.grid_w - 1), float(self.grid_h - 1)
        x_norm = self.drone_pos[0] / gx
        y_norm = self.drone_pos[1] / gy

        # Target: either current package's dest or next highest-priority package dest
        if self.carrying_package and self.current_package is not None:
            target = self.current_package.dest_pos
        else:
            # "next target" = highest priority package's destination
            remaining = [p for p in self.packages if not p.delivered]
            if remaining:
                remaining.sort(key=lambda p: (-p.priority, self._manhattan(self.drone_pos, p.dest_pos)))
                target = remaining[0].dest_pos
            else:
                target = self.wh_center

        dx_t = (target[0] - self.drone_pos[0]) / max(1.0, gx)
        dy_t = (target[1] - self.drone_pos[1]) / max(1.0, gy)
        dist_t = self._manhattan(self.drone_pos, target) / float(self.grid_w + self.grid_h)

        # Nearest charger
        nearest = min(self.chargers, key=lambda c: self._manhattan(self.drone_pos, c))
        dx_c = (nearest[0] - self.drone_pos[0]) / max(1.0, gx)
        dy_c = (nearest[1] - self.drone_pos[1]) / max(1.0, gy)
        dist_c = self._manhattan(self.drone_pos, nearest) / float(self.grid_w + self.grid_h)

        carrying = 1.0 if self.carrying_package else 0.0
        time_left = 1.0 - (self.steps / float(self.episode_step_limit))
        pkgs_left = sum(1 for p in self.packages if not p.delivered) / max(1.0, float(self.n_packages))

        obs = np.array([
            x_norm, y_norm,
            float(self.battery),
            dx_t, dy_t, float(dist_t),
            dx_c, dy_c, float(dist_c),
            carrying,
            float(time_left),
            float(pkgs_left),
        ], dtype=np.float32)

        return obs
