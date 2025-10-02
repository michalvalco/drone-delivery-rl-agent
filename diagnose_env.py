
"""
Environment Diagnostic Script
This script tests if the drone delivery environment is solvable and identifies issues.
"""

import sys
import numpy as np
from environment.drone_env import DroneDeliveryEnv


def test_random_agent(episodes=10):
    """Test with completely random actions"""
    print("=" * 60)
    print("TEST 1: Random Agent Baseline")
    print("=" * 60)
    
    env = DroneDeliveryEnv(fixed_env=True)
    total_rewards = []
    total_deliveries = []
    episode_lengths = []
    
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        deliveries = 0
        
        while not done:
            action = env.action_space.sample()  # Random action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            if 'deliveries' in info:
                deliveries = info['deliveries']
        
        total_rewards.append(episode_reward)
        total_deliveries.append(deliveries)
        episode_lengths.append(steps)
    
    print(f"Episodes: {episodes}")
    print(f"Avg Reward: {np.mean(total_rewards):.2f} (±{np.std(total_rewards):.2f})")
    print(f"Avg Deliveries: {np.mean(total_deliveries):.2f}")
    print(f"Avg Steps: {np.mean(episode_lengths):.2f}")
    print(f"Success Rate: {sum(1 for d in total_deliveries if d >= 3) / episodes * 100:.1f}%")
    print()


def test_simple_policy(episodes=5):
    """Test with a simple rule-based policy"""
    print("=" * 60)
    print("TEST 2: Simple Rule-Based Policy")
    print("=" * 60)
    print("Policy: Always move toward nearest package/destination")
    print()
    
    env = DroneDeliveryEnv(fixed_env=True)
    
    for ep in range(episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n--- Episode {ep + 1} ---")
        
        while not done and steps < 50:  # Limit to 50 steps for debugging
            # Parse state (based on documented state space)
            drone_x = state[0]
            drone_y = state[1]
            battery = state[2]
            
            # Simple policy: move toward target
            target_x = state[6]  # Target location x (approximate index)
            target_y = state[7]  # Target location y
            
            # Choose action based on direction to target
            if abs(target_x - drone_x) > abs(target_y - drone_y):
                if target_x > drone_x:
                    action = 2  # Move East
                else:
                    action = 3  # Move West
            else:
                if target_y > drone_y:
                    action = 1  # Move South
                else:
                    action = 0  # Move North
            
            # If very low battery, try to charge
            if battery < 0.2:
                action = 4  # Try to charge
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            
            if steps <= 10 or reward != -1:  # Print interesting steps
                print(f"Step {steps}: Action={action}, Reward={reward:.1f}, "
                      f"Battery={state[2]:.2f}, Info={info}")
            
            state = next_state
        
        print(f"Episode {ep + 1} Result: Reward={episode_reward:.1f}, Steps={steps}, "
              f"Deliveries={info.get('deliveries', 0)}")


def test_reward_accessibility():
    """Test if positive rewards are actually reachable"""
    print("=" * 60)
    print("TEST 3: Reward Accessibility Check")
    print("=" * 60)
    
    env = DroneDeliveryEnv(fixed_env=True)
    state, info = env.reset()
    
    print(f"Initial state shape: {state.shape}")
    print(f"Action space: {env.action_space}")
    print(f"State space: {env.observation_space}")
    print()
    
    # Test each action type
    print("Testing action rewards:")
    for action in range(5):
        env.reset()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"Action {action}: Reward = {reward:.1f}, Terminated = {terminated}, Info = {info}")
    
    print()


def test_episode_termination():
    """Check what causes episodes to end"""
    print("=" * 60)
    print("TEST 4: Episode Termination Analysis")
    print("=" * 60)
    
    env = DroneDeliveryEnv(fixed_env=True)
    
    termination_reasons = {
        'max_steps': 0,
        'battery_depleted': 0,
        'all_delivered': 0,
        'unknown': 0
    }
    
    for ep in range(20):
        state, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Determine termination reason
        if info.get('deliveries', 0) >= 3:
            termination_reasons['all_delivered'] += 1
        elif state[2] <= 0:  # Battery level
            termination_reasons['battery_depleted'] += 1
        elif steps >= 200:
            termination_reasons['max_steps'] += 1
        else:
            termination_reasons['unknown'] += 1
    
    print("Termination reasons (20 episodes):")
    for reason, count in termination_reasons.items():
        print(f"  {reason}: {count} ({count/20*100:.1f}%)")
    print()


def main():
    print("\n" + "=" * 60)
    print("DRONE DELIVERY ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    print()
    
    try:
        # Run all tests
        test_reward_accessibility()
        test_random_agent(episodes=20)
        test_episode_termination()
        test_simple_policy(episodes=3)
        
        print("\n" + "=" * 60)
        print("DIAGNOSTIC COMPLETE")
        print("=" * 60)
        print("\nRecommendations:")
        print("1. Check if random agent ever gets positive rewards")
        print("2. Verify battery doesn't deplete too quickly")
        print("3. Ensure pickup/delivery mechanics are working")
        print("4. Check if reward structure encourages the right behaviors")
        
    except Exception as e:
        print(f"\n❌ ERROR during diagnostics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
