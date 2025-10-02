"""
Training script for Drone Delivery RL Agent.

This script trains a Deep Q-Network (DQN) agent to perform autonomous drone deliveries
with battery management constraints. The agent learns to optimize delivery routes while
managing limited battery resources and utilizing charging stations.

Author: Michal Valčo
Course: AI Agents - Lesson 10: RL Agent Practical Project
"""

import argparse
import os
import numpy as np
from typing import Optional

from environment import DroneDeliveryEnv
from agent.dqn_agent import DQNAgent
from utils.logger import TrainingLogger
from utils.visualization import TrainingVisualizer


def train(
    num_episodes: int = 1000,
    grid_size: int = 20,
    max_steps: int = 250,
    num_packages: int = 3,
    num_stations: int = 4,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    buffer_capacity: int = 10000,
    target_update_freq: int = 10,
    save_freq: int = 100,
    render_freq: int = 0,
    experiment_name: Optional[str] = None
) -> None:
    """
    Train a DQN agent on the Drone Delivery environment.
    
    Args:
        num_episodes: Number of training episodes
        grid_size: Size of the square grid world
        max_steps: Maximum steps per episode before truncation
        num_packages: Number of packages to deliver per episode
        num_stations: Number of charging stations in the environment
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor for future rewards
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Multiplicative decay factor for epsilon per episode
        batch_size: Mini-batch size for training
        buffer_capacity: Maximum size of replay buffer
        target_update_freq: Frequency (in episodes) to update target network
        save_freq: Frequency (in episodes) to save model checkpoints
        render_freq: Frequency (in episodes) to render environment (0=disabled)
        experiment_name: Optional name for the experiment (used in logging)
    """
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Initialize environment
    env = DroneDeliveryEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        num_packages=num_packages,
        num_stations=num_stations,
        render_mode='human' if render_freq > 0 else None,
        fixed_env=True  # Fixed environment for consistent learning
    )
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Initialize logging and visualization
    logger = TrainingLogger(experiment_name=experiment_name)
    visualizer = TrainingVisualizer()
    
    # Log hyperparameters
    hyperparameters = {
        'num_episodes': num_episodes,
        'grid_size': grid_size,
        'max_steps': max_steps,
        'num_packages': num_packages,
        'num_stations': num_stations,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'epsilon_decay': epsilon_decay,
        'batch_size': batch_size,
        'buffer_capacity': buffer_capacity,
        'target_update_freq': target_update_freq
    }
    logger.log_hyperparameters(hyperparameters)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    episode_deliveries = []
    success_count = 0
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Environment: {grid_size}x{grid_size} grid, {num_packages} packages, {num_stations} charging stations")
    print(f"Agent: DQN with epsilon={epsilon_start:.2f} → {epsilon_end:.2f}")
    print(f"Device: {agent.device}")
    print("-" * 80)
    
    # Main training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        loss_count = 0
        
        done = False
        truncated = False
        
        # Episode loop
        while not (done or truncated):
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done or truncated)
            
            # Train agent (if enough samples in buffer)
            loss = agent.train_step()
            if loss > 0:
                episode_loss += loss
                loss_count += 1
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            # Render if requested
            if render_freq > 0 and episode % render_freq == 0:
                env.render()
        
        # End of episode updates
        agent.end_episode()
        
        # Track success rate
        if info['delivered_count'] == num_packages:
            success_count += 1
        
        # Calculate average loss for episode
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        
        # Store metrics
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_loss)
        episode_deliveries.append(info['delivered_count'])
        
        # Log episode data
        logger.log_episode(episode, {
            'total_reward': episode_reward,
            'loss': avg_loss,
            'delivered_count': info['delivered_count'],
            'steps': steps,
            'battery': info['battery'],
            'epsilon': agent.epsilon
        })
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_deliveries = np.mean(episode_deliveries[-10:])
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Deliveries: {avg_deliveries:.2f} | "
                  f"Success Rate: {success_rate:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(f"models/dqn_agent_ep{episode + 1}.pth")
            print(f"  → Model checkpoint saved at episode {episode + 1}")
    
    # Cleanup
    env.close()
    
    # Save final model
    agent.save("models/dqn_agent_final.pth")
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Final model saved to: models/dqn_agent_final.pth")
    print(f"Final success rate: {success_count / num_episodes * 100:.1f}%")
    print("=" * 80)
    
    # Plot training metrics
    visualizer.plot_training_metrics(
        episode_rewards,
        episode_losses,
        episode_deliveries
    )
    
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train DQN agent on Drone Delivery environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Environment parameters
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    parser.add_argument('--grid-size', type=int, default=20, 
                       help='Size of the grid')
    parser.add_argument('--max-steps', type=int, default=250, 
                       help='Maximum steps per episode')
    parser.add_argument('--packages', type=int, default=3, 
                       help='Number of packages')
    parser.add_argument('--stations', type=int, default=4, 
                       help='Number of charging stations')
    
    # Agent hyperparameters
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, 
                       help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0, 
                       help='Initial epsilon')
    parser.add_argument('--epsilon-end', type=float, default=0.01, 
                       help='Final epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, 
                       help='Epsilon decay rate per episode')
    parser.add_argument('--batch-size', type=int, default=64, 
                       help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=10000, 
                       help='Replay buffer capacity')
    parser.add_argument('--target-update', type=int, default=10, 
                       help='Target network update frequency (episodes)')
    
    # Training options
    parser.add_argument('--save-freq', type=int, default=100, 
                       help='Model save frequency (episodes)')
    parser.add_argument('--render-freq', type=int, default=0, 
                       help='Rendering frequency (episodes, 0=disabled)')
    parser.add_argument('--experiment', type=str, default=None, 
                       help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Run training
    train(
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        num_packages=args.packages,
        num_stations=args.stations,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        target_update_freq=args.target_update,
        save_freq=args.save_freq,
        render_freq=args.render_freq,
        experiment_name=args.experiment
    )