"""
train.py
- Unified training script for DQN, Double DQN, REINFORCE, A2C, PPO
- Logs episode rewards, success rate, losses
- Saves checkpoints and training curves

Usage:
    python train.py --algo dqn --total_steps 300000
    python train.py --algo double_dqn --total_steps 300000
    python train.py --algo reinforce --total_steps 300000
    python train.py --algo a2c --total_steps 300000
    python train.py --algo ppo --total_steps 300000
"""

import os
import argparse
import numpy as np
import json
from collections import deque

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.parking_env import ParkingEnv
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.reinforce import REINFORCEAgent
from agents.a2c import A2CAgent
from agents.ppo import PPOAgent


# ------------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------------

class TrainingLogger:
    def __init__(self, log_dir: str, algo: str):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{algo}_log.json")
        self.data = {
            "episode_rewards": [],
            "episode_lengths": [],
            "success_rate":    [],
            "collision_rate":  [],
            "time_over_rate":  [],
            "losses":          [],
            "steps":           [],
        }
        self._recent_success = deque(maxlen=100)
        self._recent_collision = deque(maxlen=100)
        self._recent_time_over = deque(maxlen=100)

    def log_episode(self, step, ep_reward, ep_length, success, collided, time_over):
        self._recent_success.append(float(success))
        self._recent_collision.append(float(collided))
        self._recent_time_over.append(float(time_over))
        self.data["episode_rewards"].append(ep_reward)
        self.data["episode_lengths"].append(ep_length)
        self.data["success_rate"].append(np.mean(self._recent_success))
        self.data["collision_rate"].append(np.mean(self._recent_collision))
        self.data["time_over_rate"].append(np.mean(self._recent_time_over))
        self.data["steps"].append(step)

    def log_loss(self, loss_dict):
        self.data["losses"].append(loss_dict)

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def print_status(self, step, total_steps, ep_reward, ep_length):
        sr = self.data["success_rate"][-1] if self.data["success_rate"] else 0.0
        cr = self.data["collision_rate"][-1] if self.data["collision_rate"] else 0.0
        tr = self.data["time_over_rate"][-1] if self.data["time_over_rate"] else 0.0
        print(
            f"[{step:>8d}/{total_steps}] "
            f"reward={ep_reward:>8.2f}  "
            f"len={ep_length:>4d}  "
            f"success={sr:.2%}  "
            f"collision={cr:.2%}  "
            f"time_over={tr:.2%}"
        )


# ------------------------------------------------------------------
# Training loops (one per algorithm family)
# ------------------------------------------------------------------

def train_dqn(env, agent, total_steps, log_dir, algo, save_every):
    logger = TrainingLogger(log_dir, algo)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    obs, _ = env.reset()
    ep_reward, ep_length = 0.0, 0
    episode = 0

    for step in range(1, total_steps + 1):
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        loss = agent.update()

        ep_reward += reward
        ep_length += 1
        obs = next_obs

        if done:
            success = env.is_success(obs, info)
            collided = bool(info.get("crashed", False))
            time_over = bool(truncated and not success and not collided)
            logger.log_episode(step, ep_reward, ep_length, success, collided, time_over)
            if loss is not None:
                logger.log_loss({"loss": loss})

            if episode % 20 == 0:
                logger.print_status(step, total_steps, ep_reward, ep_length)

            ep_reward, ep_length = 0.0, 0
            episode += 1
            obs, _ = env.reset()

        if step % save_every == 0:
            agent.save(os.path.join(ckpt_dir, f"{algo}_step{step}.pt"))
            logger.save()

    agent.save(os.path.join(ckpt_dir, f"{algo}_final.pt"))
    logger.save()
    print(f"\n[{algo.upper()}] Training done. Log saved to {logger.path}")


def train_onpolicy(env, agent, total_steps, log_dir, algo, save_every):
    """Shared training loop for A2C and PPO (on-policy)."""
    logger = TrainingLogger(log_dir, algo)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    obs, _ = env.reset()
    ep_reward, ep_length = 0.0, 0
    episode = 0

    for step in range(1, total_steps + 1):
        action, log_prob, value = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, log_prob, value, done)
        obs = next_obs
        ep_reward += reward
        ep_length += 1

        if done:
            success = env.is_success(obs, info)
            collided = bool(info.get("crashed", False))
            time_over = bool(truncated and not success and not collided)
            logger.log_episode(step, ep_reward, ep_length, success, collided, time_over)

            if episode % 20 == 0:
                logger.print_status(step, total_steps, ep_reward, ep_length)

            ep_reward, ep_length = 0.0, 0
            episode += 1
            obs, _ = env.reset()

        if agent.should_update():
            metrics = agent.update(obs, done)
            if metrics:
                logger.log_loss(metrics)

        if step % save_every == 0:
            agent.save(os.path.join(ckpt_dir, f"{algo}_step{step}.pt"))
            logger.save()

    agent.save(os.path.join(ckpt_dir, f"{algo}_final.pt"))
    logger.save()
    print(f"\n[{algo.upper()}] Training done. Log saved to {logger.path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",        type=str,   default="ppo",
                        choices=["dqn", "double_dqn", "reinforce", "a2c", "ppo"])
    parser.add_argument("--total_steps", type=int,   default=300_000)
    parser.add_argument("--noise_std",   type=float, default=0.02)
    parser.add_argument("--n_vehicles",  type=int,   default=6)
    parser.add_argument("--log_dir",     type=str,   default="logs")
    parser.add_argument("--save_every",  type=int,   default=50_000)
    parser.add_argument("--device",      type=str,   default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Algorithm : {args.algo.upper()}")
    print(f"  Steps     : {args.total_steps:,}")
    print(f"  Device    : {args.device}")
    print(f"{'='*50}\n")

    # Build environment
    discrete = (args.algo in ["dqn", "double_dqn"])
    env = ParkingEnv(
        discrete=discrete,
        noise_std=args.noise_std,
        n_other_vehicles=args.n_vehicles,
    )

    obs_dim = env.observation_space.shape[0]

    # Build agent
    if args.algo == "dqn":
        agent = DQNAgent(
            obs_dim=obs_dim,
            n_actions=env.N_DISCRETE_ACTIONS,
            device=args.device,
        )
        train_dqn(env, agent, args.total_steps, args.log_dir, args.algo, args.save_every)

    elif args.algo == "double_dqn":
        agent = DoubleDQNAgent(
            obs_dim=obs_dim,
            n_actions=env.N_DISCRETE_ACTIONS,
            device=args.device,
        )
        train_dqn(env, agent, args.total_steps, args.log_dir, args.algo, args.save_every)

    elif args.algo == "reinforce":
        agent = REINFORCEAgent(
            obs_dim=obs_dim,
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
        train_onpolicy(env, agent, args.total_steps, args.log_dir, args.algo, args.save_every)

    elif args.algo == "a2c":
        agent = A2CAgent(
            obs_dim=obs_dim,
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
        train_onpolicy(env, agent, args.total_steps, args.log_dir, args.algo, args.save_every)

    elif args.algo == "ppo":
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=env.action_space.shape[0],
            device=args.device,
        )
        train_onpolicy(env, agent, args.total_steps, args.log_dir, args.algo, args.save_every)

    env.close()


if __name__ == "__main__":
    main()
