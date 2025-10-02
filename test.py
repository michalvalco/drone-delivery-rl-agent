import pytest
import numpy as np
from environment.drone_env import DroneDeliveryEnv
from environment.entities import Package, ChargingStation


class TestDroneDeliveryEnv:
    
    @pytest.fixture
    def env(self):
        return DroneDeliveryEnv(
            grid_size=10,
            max_steps=100,
            num_packages=2,
            num_stations=2,
            render_mode=None
        )
    
    def test_initialization(self, env):
        assert env.grid_size == 10
        assert env.max_steps == 100
        assert env.num_packages == 2
        assert env.num_stations == 2
        assert env.observation_space.shape == (12,)
        assert env.action_space.n == 5
    
    def test_reset(self, env):
        observation, info = env.reset(seed=42)
        
        assert observation.shape == (12,)
        assert np.all(observation >= 0.0) and np.all(observation <= 1.0)
        
        assert len(env.packages) == 2
        assert len(env.charging_stations) == 2
        assert env.battery == 1.0
        assert env.current_step == 0
        assert env.delivered_count == 0
        assert not env.carrying_package
        
        assert 'drone_position' in info
        assert 'battery' in info
        assert 'delivered_count' in info
    
    def test_movement_actions(self, env):
        env.reset(seed=42)
        initial_pos = env.drone_pos.copy()
        initial_battery = env.battery
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert env.battery < initial_battery
        assert reward == -1
        assert not terminated
        assert not truncated
    
    def test_battery_depletion(self, env):
        env.reset(seed=42)
        
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated:
                break
        
        assert env.battery <= 0
        assert terminated
    
    def test_package_pickup(self, env):
        env.reset(seed=42)
        
        pkg = env.packages[0]
        env.drone_pos = pkg.position.copy()
        
        obs, reward, terminated, truncated, info = env.step(4)
        
        assert pkg.picked_up
        assert env.carrying_package
        assert reward == 10
    
    def test_package_delivery(self, env):
        env.reset(seed=42)
        
        pkg = env.packages[0]
        pkg.picked_up = True
        env.carrying_package = True
        env.drone_pos = pkg.destination.copy()
        
        initial_delivered = env.delivered_count
        
        obs, reward, terminated, truncated, info = env.step(4)
        
        assert pkg.delivered
        assert not env.carrying_package
        assert env.delivered_count == initial_delivered + 1
        assert reward == 100 * pkg.priority
    
    def test_charging(self, env):
        env.reset(seed=42)
        
        env.battery = 0.5
        station = env.charging_stations[0]
        env.drone_pos = station.position.copy()
        
        initial_battery = env.battery
        
        obs, reward, terminated, truncated, info = env.step(4)
        
        assert env.battery > initial_battery
        assert reward == 5
    
    def test_invalid_interact_action(self, env):
        env.reset(seed=42)
        
        env.drone_pos = np.array([5, 5])
        
        obs, reward, terminated, truncated, info = env.step(4)
        
        assert reward == -5
    
    def test_all_packages_delivered(self, env):
        env.reset(seed=42)
        
        for pkg in env.packages:
            pkg.delivered = True
        env.delivered_count = len(env.packages)
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert terminated
    
    def test_max_steps_truncation(self, env):
        env.reset(seed=42)
        env.current_step = env.max_steps - 1
        
        obs, reward, terminated, truncated, info = env.step(0)
        
        assert truncated
        if env.delivered_count < env.num_packages:
            assert reward <= -1
    
    def test_observation_bounds(self, env):
        env.reset(seed=42)
        
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
            
            if terminated or truncated:
                break
    
    def test_deterministic_reset(self):
        env1 = DroneDeliveryEnv(grid_size=10, num_packages=2, num_stations=2)
        env2 = DroneDeliveryEnv(grid_size=10, num_packages=2, num_stations=2)
        
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(env1.drone_pos, env2.drone_pos)


class TestPackage:
    
    def test_package_creation(self):
        pkg = Package(
            position=np.array([5, 5]),
            destination=np.array([10, 10]),
            priority=2
        )
        
        assert np.array_equal(pkg.position, np.array([5, 5]))
        assert np.array_equal(pkg.destination, np.array([10, 10]))
        assert pkg.priority == 2
        assert not pkg.picked_up
        assert not pkg.delivered


class TestChargingStation:
    
    def test_station_creation(self):
        station = ChargingStation(
            position=np.array([3, 7]),
            charging_rate=0.15
        )
        
        assert np.array_equal(station.position, np.array([3, 7]))
        assert station.charging_rate == 0.15


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

