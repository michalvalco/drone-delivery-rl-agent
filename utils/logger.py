import csv
import os
from datetime import datetime
from typing import Dict, List, Any


class TrainingLogger:
    def __init__(self, log_dir: str = "logs", experiment_name: str = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.csv")
        
        self.episode_data = []
        self.headers = None
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        log_entry = {"episode": episode, **metrics}
        
        if self.headers is None:
            self.headers = list(log_entry.keys())
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
        
        self.episode_data.append(log_entry)
        
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(log_entry)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        hp_file = os.path.join(self.log_dir, f"{self.experiment_name}_hyperparameters.txt")
        
        with open(hp_file, 'w') as f:
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\nHyperparameters:\n")
            f.write("-" * 50 + "\n")
            for key, value in hyperparameters.items():
                f.write(f"{key}: {value}\n")
    
    def get_episode_data(self) -> List[Dict[str, Any]]:
        return self.episode_data
    
    def get_metrics_history(self, metric_name: str) -> List[float]:
        return [episode[metric_name] for episode in self.episode_data if metric_name in episode]
    
    def print_summary(self):
        if not self.episode_data:
            print("No episode data logged yet.")
            return
        
        print(f"\n{'='*60}")
        print(f"Training Summary - {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total Episodes: {len(self.episode_data)}")
        
        if 'total_reward' in self.episode_data[0]:
            rewards = self.get_metrics_history('total_reward')
            print(f"\nRewards:")
            print(f"  Mean: {sum(rewards)/len(rewards):.2f}")
            print(f"  Min: {min(rewards):.2f}")
            print(f"  Max: {max(rewards):.2f}")
            print(f"  Last 100 avg: {sum(rewards[-100:])/min(100, len(rewards)):.2f}")
        
        if 'delivered_count' in self.episode_data[0]:
            deliveries = self.get_metrics_history('delivered_count')
            print(f"\nDeliveries:")
            print(f"  Mean: {sum(deliveries)/len(deliveries):.2f}")
            print(f"  Success Rate: {sum(1 for d in deliveries if d > 0)/len(deliveries)*100:.1f}%")
        
        if 'steps' in self.episode_data[0]:
            steps = self.get_metrics_history('steps')
            print(f"\nSteps:")
            print(f"  Mean: {sum(steps)/len(steps):.1f}")
        
        if 'loss' in self.episode_data[0]:
            losses = [l for l in self.get_metrics_history('loss') if l > 0]
            if losses:
                print(f"\nLoss:")
                print(f"  Mean: {sum(losses)/len(losses):.4f}")
        
        print(f"{'='*60}\n")
    
    def close(self):
        self.print_summary()
        print(f"Log saved to: {self.log_file}")

