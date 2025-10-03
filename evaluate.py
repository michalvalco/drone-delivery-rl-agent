# evaluate.py
"""
Evaluate a trained Drone Delivery RL agent.

- Loads a saved DQN checkpoint.
- Runs N evaluation episodes.
- By default uses greedy policy (epsilon=0.0); you can override with --epsilon 0.05
  to observe varied behavior and verify carrying/delivery visually.

Author: Michal Valčo
Course: AI Agents – Lesson 10
"""

import argparse
import os
import time
import numpy as np
import torch

from environment.drone_env import DroneDeliveryEnv
from agent.dqn_agent import DQNAgent


def evaluate(
    model_path: str,
    episodes: int = 10,
    render: bool = True,
    epsilon: float | None = None,
    fixed_env: bool = True,
    base_seed: int = 42,
    max_steps: int | None = None,
) -> None:
    # Create env with the same deterministic layout you trained on (by default)
    env = DroneDeliveryEnv(fixed_env=fixed_env, base_seed=base_seed)
    obs, _ = env.reset(seed=base_seed)

    state_dim = int(np.array(obs, dtype=np.float32).shape[0])
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=0.0,    # will be overwritten below
        epsilon_end=0.0,
        epsilon_decay=1.0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    agent.load(model_path)

    # Greedy by default; allow optional exploration for demo/debug
    if epsilon is None:
        agent.epsilon = 0.0
    else:
        agent.epsilon = float(epsilon)

    print(f"[Eval] Loaded {model_path}")
    print(f"[Eval] Episodes={episodes}  epsilon={agent.epsilon}  fixed_env={fixed_env}  seed={base_seed}")

    returns = []
    deliveries = []
    charges = []
    steps_list = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=base_seed if fixed_env else None)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_deliveries = 0
        ep_charges = 0
        steps = 0

        while not (done or truncated):
            action = agent.select_action(np.array(obs, dtype=np.float32))
            next_obs, reward, done, truncated, info = env.step(action)

            ep_ret += float(reward)
            steps += 1

            # book-keep some common events if env exposes them in info dict
            if isinstance(info, dict):
                ep_deliveries += int(info.get("delivered_now", 0))
                ep_charges += int(info.get("charged_now", 0))

            if render:
                env.render()
                # tiny sleep for visibility; tune as needed
                time.sleep(0.02)

            obs = next_obs
            if max_steps is not None and steps >= max_steps:
                break

        returns.append(ep_ret)
        deliveries.append(ep_deliveries)
        charges.append(ep_charges)
        steps_list.append(steps)
        print(f"[Eval] Ep {ep:03d} | return={ep_ret:7.2f} | deliveries={ep_deliveries} | charges={ep_charges} | steps={steps}")

    print("\n[Eval] Summary")
    print(f"  episodes:  {episodes}")
    print(f"  return:    mean={np.mean(returns):.2f}  std={np.std(returns):.2f}")
    print(f"  deliveries mean={np.mean(deliveries):.2f}")
    print(f"  charges:   mean={np.mean(charges):.2f}")
    print(f"  steps:     mean={np.mean(steps_list):.1f}")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN on DroneDeliveryEnv")
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.pth)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--epsilon", type=float, default=None, help="Optional epsilon during eval, e.g. 0.05")
    parser.add_argument("--fixed-env", action="store_true", help="Use fixed map across episodes (default on)", default=True)
    parser.add_argument("--base-seed", type=int, default=42, help="Seed used for the fixed map")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap per episode during eval")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        episodes=args.episodes,
        render=not args.no_render,
        epsilon=args.epsilon,
        fixed_env=args.fixed_env,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
