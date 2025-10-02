"""
Evaluation script for trained Drone Delivery RL Agent.

This script loads a trained DQN model and evaluates its performance across multiple
episodes, providing detailed statistics and optionally rendering the agent's behavior.

Author: Michal Valčo
Course: AI Agents - Lesson 10: RL Agent Practical Project
"""

import argparse
import numpy as np
from typing import Optional

from environment import DroneDeliveryEnv
from agent.dqn_agent import DQNAgent


def evaluate(
    model_path: str,
    num_episodes: int = 10,
    grid_size: int = 20,
    max_steps: int = 250,
    num_packages: int = 3,
    num_stations: int = 4,
    render: bool = True,
    detailed_output: bool = False
) -> None:
    """
    Evaluate a trained DQN agent on the Drone Delivery environment.
    
    Args:
        model_path: Path to the saved model checkpoint
        num_episodes: Number of episodes to evaluate
        grid_size: Size of the square grid world
        max_steps: Maximum steps per episode
        num_packages: Number of packages to deliver per episode
        num_stations: Number of charging stations
        render: Whether to render the environment visually
        detailed_output: Whether to print detailed step-by-step information
    """
    # Initialize environment
    env = DroneDeliveryEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        num_packages=num_packages,
        num_stations=num_stations,
        render_mode='human' if render else None
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Load trained model
    try:
        agent.load(model_path)
        agent.epsilon = 0.0  # Disable exploration for evaluation
        print(f"✓ Successfully loaded model from: {model_path}")
    except FileNotFoundError:
        print(f"✗ Error: Model file not found at {model_path}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print(f"Evaluating for {num_episodes} episodes...")
    print(f"Device: {agent.device}")
    print("-" * 80)
    
    # Evaluation metrics
    total_rewards = []
    total_deliveries = []
    total_steps = []
    battery_at_end = []
    success_count = 0
    
    # Evaluation loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action (greedy policy, no exploration)
            action = agent.select_action(state, training=False)
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            
            if detailed_output:
                print(f"  Step {steps + 1}: Action={action}, Reward={reward:.2f}, Battery={info['battery']:.2f}")
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            # Render if requested
            if render:
                env.render()
        
        # Collect episode statistics
        delivered = info['delivered_count']
        battery = info['battery']
        
        total_rewards.append(episode_reward)
        total_deliveries.append(delivered)
        total_steps.append(steps)
        battery_at_end.append(battery)
        
        if delivered == num_packages:
            success_count += 1
        
        # Print episode summary
        status = '✓ SUCCESS' if delivered == num_packages else '✗ INCOMPLETE'
        print(f"Episode {episode + 1}/{num_episodes}: {status}")
        print(f"  Reward: {episode_reward:.2f} | Delivered: {delivered}/{num_packages} | "
              f"Steps: {steps} | Battery: {battery:.2f}")
        print("-" * 80)
    
    env.close()
    
    # Print comprehensive evaluation summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    
    print(f"\n📊 Rewards:")
    print(f"  Mean:   {np.mean(total_rewards):>8.2f}")
    print(f"  Std:    {np.std(total_rewards):>8.2f}")
    print(f"  Min:    {np.min(total_rewards):>8.2f}")
    print(f"  Max:    {np.max(total_rewards):>8.2f}")
    print(f"  Median: {np.median(total_rewards):>8.2f}")
    
    print(f"\n📦 Deliveries:")
    print(f"  Mean:         {np.mean(total_deliveries):>8.2f}")
    print(f"  Success Rate: {success_count / num_episodes * 100:>8.1f}%")
    print(f"  Total Pkg:    {sum(total_deliveries):>8d}/{num_episodes * num_packages}")
    
    print(f"\n👣 Steps:")
    print(f"  Mean:   {np.mean(total_steps):>8.1f}")
    print(f"  Std:    {np.std(total_steps):>8.1f}")
    print(f"  Min:    {np.min(total_steps):>8d}")
    print(f"  Max:    {np.max(total_steps):>8d}")
    
    print(f"\n🔋 Battery (at episode end):")
    print(f"  Mean:   {np.mean(battery_at_end):>8.2f}")
    print(f"  Std:    {np.std(battery_at_end):>8.2f}")
    print(f"  Min:    {np.min(battery_at_end):>8.2f}")
    
    print("\n" + "=" * 80)
    
    # Performance grading
    success_rate = success_count / num_episodes * 100
    if success_rate >= 90:
        grade = "A+ Excellent"
    elif success_rate >= 80:
        grade = "A  Very Good"
    elif success_rate >= 70:
        grade = "B  Good"
    elif success_rate >= 60:
        grade = "C  Acceptable"
    else:
        grade = "D  Needs Improvement"
    
    print(f"Performance Grade: {grade}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate trained DQN agent on Drone Delivery task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model file (.pth)')
    
    # Evaluation parameters
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Number of evaluation episodes')
    
    # Environment parameters
    parser.add_argument('--grid-size', type=int, default=20, 
                       help='Size of the grid')
    parser.add_argument('--max-steps', type=int, default=250, 
                       help='Maximum steps per episode')
    parser.add_argument('--packages', type=int, default=3, 
                       help='Number of packages')
    parser.add_argument('--stations', type=int, default=4, 
                       help='Number of charging stations')
    
    # Display options
    parser.add_argument('--no-render', action='store_true', 
                       help='Disable visual rendering')
    parser.add_argument('--detailed', action='store_true', 
                       help='Print detailed step-by-step information')
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        num_packages=args.packages,
        num_stations=args.stations,
        render=not args.no_render,
        detailed_output=args.detailed
    )