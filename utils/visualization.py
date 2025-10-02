import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


class TrainingVisualizer:
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_training_metrics(
        self,
        episode_rewards: List[float],
        episode_losses: List[float],
        episode_deliveries: List[int],
        window_size: int = 50
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Over Episodes', fontsize=16, fontweight='bold')
        
        episodes = np.arange(len(episode_rewards))
        
        axes[0, 0].plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
        if len(episode_rewards) >= window_size:
            smoothed_rewards = self._moving_average(episode_rewards, window_size)
            axes[0, 0].plot(
                episodes[window_size-1:],
                smoothed_rewards,
                color='darkblue',
                linewidth=2,
                label=f'MA({window_size})'
            )
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(episodes, episode_losses, alpha=0.3, color='red', label='Raw')
        if len(episode_losses) >= window_size:
            smoothed_losses = self._moving_average(episode_losses, window_size)
            axes[0, 1].plot(
                episodes[window_size-1:],
                smoothed_losses,
                color='darkred',
                linewidth=2,
                label=f'MA({window_size})'
            )
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(episodes, episode_deliveries, alpha=0.3, color='green', label='Raw')
        if len(episode_deliveries) >= window_size:
            smoothed_deliveries = self._moving_average(episode_deliveries, window_size)
            axes[1, 0].plot(
                episodes[window_size-1:],
                smoothed_deliveries,
                color='darkgreen',
                linewidth=2,
                label=f'MA({window_size})'
            )
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Packages Delivered')
        axes[1, 0].set_title('Delivery Success Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(episode_rewards) >= window_size:
            recent_window = min(200, len(episode_rewards))
            recent_rewards = episode_rewards[-recent_window:]
            recent_episodes = episodes[-recent_window:]
            
            axes[1, 1].plot(recent_episodes, recent_rewards, color='purple', linewidth=1.5)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Total Reward')
            axes[1, 1].set_title(f'Recent Performance (Last {recent_window} Episodes)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_metrics.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training metrics plot to {save_path}")
        plt.close()
    
    def plot_comparison(
        self,
        metrics_dict: Dict[str, List[float]],
        metric_name: str,
        title: str,
        ylabel: str,
        window_size: int = 50
    ):
        plt.figure(figsize=(12, 6))
        
        for agent_name, values in metrics_dict.items():
            episodes = np.arange(len(values))
            
            if len(values) >= window_size:
                smoothed = self._moving_average(values, window_size)
                plt.plot(
                    episodes[window_size-1:],
                    smoothed,
                    linewidth=2,
                    label=agent_name
                )
            else:
                plt.plot(episodes, values, linewidth=2, label=agent_name)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'{metric_name}_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
        plt.close()
    
    def plot_episode_details(
        self,
        episode_steps: List[Dict],
        episode_num: int
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Episode {episode_num} Details', fontsize=16, fontweight='bold')
        
        steps = np.arange(len(episode_steps))
        battery_levels = [step['battery'] for step in episode_steps]
        rewards = [step['reward'] for step in episode_steps]
        
        axes[0, 0].plot(steps, battery_levels, color='orange', linewidth=2)
        axes[0, 0].axhline(y=0.2, color='red', linestyle='--', label='Critical Level')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Battery Level')
        axes[0, 0].set_title('Battery Consumption')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(steps, rewards, color='green', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Step Rewards')
        axes[0, 1].grid(True, alpha=0.3)
        
        x_positions = [step['position'][0] for step in episode_steps]
        y_positions = [step['position'][1] for step in episode_steps]
        
        scatter = axes[1, 0].scatter(
            x_positions,
            y_positions,
            c=steps,
            cmap='viridis',
            s=20,
            alpha=0.6
        )
        axes[1, 0].plot(x_positions, y_positions, 'b-', alpha=0.2, linewidth=0.5)
        axes[1, 0].scatter(x_positions[0], y_positions[0], c='green', s=100, marker='o', label='Start')
        axes[1, 0].scatter(x_positions[-1], y_positions[-1], c='red', s=100, marker='X', label='End')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Y Position')
        axes[1, 0].set_title('Drone Trajectory')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Step')
        
        actions = [step['action'] for step in episode_steps]
        action_names = ['Up', 'Down', 'Right', 'Left', 'Interact']
        action_counts = [actions.count(i) for i in range(5)]
        
        axes[1, 1].bar(action_names, action_counts, color=['blue', 'cyan', 'green', 'yellow', 'red'])
        axes[1, 1].set_xlabel('Action')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Action Distribution')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'episode_{episode_num}_details.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved episode details plot to {save_path}")
        plt.close()
    
    def _moving_average(self, data: List[float], window_size: int) -> np.ndarray:
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def plot_hyperparameter_comparison(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metric_name: str = 'rewards'
    ):
        plt.figure(figsize=(14, 8))
        
        for config_name, metrics in results.items():
            if metric_name in metrics:
                episodes = np.arange(len(metrics[metric_name]))
                smoothed = self._moving_average(metrics[metric_name], 50)
                plt.plot(
                    episodes[49:],
                    smoothed,
                    linewidth=2,
                    label=config_name,
                    alpha=0.8
                )
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Hyperparameter Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9, loc='best')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'hyperparameter_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved hyperparameter comparison to {save_path}")
        plt.close()

