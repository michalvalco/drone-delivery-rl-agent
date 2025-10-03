# train.py
"""
Train a DQN agent for Drone Delivery with battery management.

Key defaults:
- fixed_env=True with base_seed=42 so the agent **learns one map** (your main concern).
- max_steps set by the env (250 as per README).
- epsilon decays slowly so we see real exploration early.

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
from agent.replay_buffer import ReplayBuffer


def train(
    episodes: int = 500,
    buffer_capacity: int = 50_000,
    batch_size: int = 64,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,  # slower decay aids discovery (charging, pickup)
    target_update_freq: int = 10,  # update target net every N episodes
    save_freq: int = 50,
    out_dir: str = "models",
    render_freq: int | None = None,
    fixed_env: bool = True,
    base_seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    env = DroneDeliveryEnv(fixed_env=fixed_env, base_seed=base_seed)
    obs, _ = env.reset(seed=base_seed)
    state_dim = int(np.array(obs, dtype=np.float32).shape[0])
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=gamma,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        device=device,
    )

    buffer = ReplayBuffer(capacity=buffer_capacity)
    global_step = 0

    print(f"[Train] episodes={episodes}  device={device}  fixed_env={fixed_env} seed={base_seed}")
    print(f"[Train] replay_capacity={buffer_capacity}  batch_size={batch_size}")
    print(f"[Train] epsilon: start={epsilon_start} end={epsilon_end} decay={epsilon_decay}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=base_seed if fixed_env else None)
        done = False
        truncated = False
        ep_ret = 0.0
        ep_loss_accum = 0.0
        ep_updates = 0
        ep_deliveries = 0
        ep_charges = 0
        steps = 0

        render_this = (render_freq is not None) and (ep % render_freq == 0)

        while not (done or truncated):
            action = agent.select_action(np.array(obs, dtype=np.float32))
            next_obs, reward, done, truncated, info = env.step(action)

            buffer.push(obs, action, reward, next_obs, done or truncated)
            obs = next_obs
            ep_ret += float(reward)
            steps += 1
            global_step += 1

            # optional book-keeping
            if isinstance(info, dict):
                ep_deliveries += int(info.get("delivered_now", 0))
                ep_charges += int(info.get("charged_now", 0))

            # Learn
            if len(buffer) >= batch_size:
                loss = agent.update(buffer.sample(batch_size))
                ep_loss_accum += float(loss)
                ep_updates += 1

            if render_this:
                env.render()
                time.sleep(0.01)

        # hard update target net periodically
        if ep % target_update_freq == 0:
            agent.hard_update()

        avg_loss = (ep_loss_accum / max(ep_updates, 1))
        print(
            f"[Train] Ep {ep:03d} | return={ep_ret:7.2f} | steps={steps:3d} | deliveries={ep_deliveries:2d} "
            f"| charges={ep_charges:2d} | loss={avg_loss:7.5f} | epsilon={agent.epsilon:.3f}"
        )

        # Save checkpoints
        if ep % save_freq == 0:
            ckpt = os.path.join(out_dir, f"dqn_ep{ep}.pth")
            agent.save(ckpt)
            print(f"[Train] Saved checkpoint -> {ckpt}")

        agent.decay_epsilon()

    # Final save
    final_path = os.path.join(out_dir, "dqn_agent_final.pth")
    agent.save(final_path)
    print(f"[Train] Saved final model -> {final_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train DQN on DroneDeliveryEnv")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.997)
    parser.add_argument("--target-update-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--render-freq", type=int, default=None, help="Render every N episodes")
    parser.add_argument("--fixed-env", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        save_freq=args.save_freq,
        out_dir=args.out_dir,
        render_freq=args.render_freq,
        fixed_env=args.fixed_env,
        base_seed=args.base_seed,
    )


if __name__ == "__main__":
    main()
