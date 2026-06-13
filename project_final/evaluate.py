"""
evaluate.py
- Load trained checkpoints and evaluate performance
- Generate comparison plots for presentation
  1. Training curves (episode reward)
  2. Success rate curves
  3. Algorithm comparison bar chart

Usage:
    # Evaluate single algorithm
    python evaluate.py --algo ppo --ckpt logs/checkpoints/ppo_final.pt

    # Compare all three algorithms (needs all logs)
    python evaluate.py --compare --log_dir logs
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.parking_env import ParkingEnv
from agents.dqn import DQNAgent
from agents.a2c import A2CAgent
from agents.ppo import PPOAgent


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_agent(env, agent, algo: str, n_episodes: int = 100):
    rewards, lengths, successes = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward, ep_length = 0.0, 0
        done = False

        while not done:
            if algo == "dqn":
                action = agent.select_action(obs, training=False)
                log_prob, value = None, None
            else:
                action, log_prob, value = agent.select_action(obs, training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_length += 1

        success = env.is_success(obs, info)
        rewards.append(ep_reward)
        lengths.append(ep_length)
        successes.append(float(success))

    return {
        "mean_reward":  np.mean(rewards),
        "std_reward":   np.std(rewards),
        "mean_length":  np.mean(lengths),
        "success_rate": np.mean(successes),
        "rewards":      rewards,
        "successes":    successes,
    }


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

COLORS = {"dqn": "#E63946", "a2c": "#457B9D", "ppo": "#2A9D8F"}
LABELS = {"dqn": "DQN (Baseline 1)", "a2c": "A2C (Baseline 2)", "ppo": "PPO (Ours)"}


def smooth(values, window: int = 20):
    """Moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(log_dir: str, save_dir: str):
    """Plot episode reward and success rate for all available algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — Autonomous Parking", fontsize=14, fontweight="bold")

    for algo in ["dqn", "a2c", "ppo"]:
        log_path = os.path.join(log_dir, f"{algo}_log.json")
        if not os.path.exists(log_path):
            continue

        with open(log_path) as f:
            data = json.load(f)

        steps   = np.array(data["steps"])
        rewards = np.array(data["episode_rewards"])
        sr      = np.array(data["success_rate"])

        color = COLORS[algo]
        label = LABELS[algo]

        # Episode reward
        smoothed_r = smooth(rewards)
        axes[0].plot(steps[:len(smoothed_r)], smoothed_r, color=color, label=label, linewidth=2)
        axes[0].fill_between(
            steps[:len(smoothed_r)],
            smoothed_r - np.std(rewards) * 0.3,
            smoothed_r + np.std(rewards) * 0.3,
            alpha=0.15, color=color,
        )

        # Success rate
        smoothed_sr = smooth(sr)
        axes[1].plot(steps[:len(smoothed_sr)], smoothed_sr * 100, color=color, label=label, linewidth=2)

    axes[0].set_xlabel("Environment Steps")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Episode Reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Environment Steps")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Success Rate (100-ep rolling avg)")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_comparison_bar(results: dict, save_dir: str):
    """Bar chart comparing final performance of all algorithms."""
    algos  = list(results.keys())
    labels = [LABELS[a] for a in algos]
    colors = [COLORS[a] for a in algos]

    mean_r = [results[a]["mean_reward"]  for a in algos]
    std_r  = [results[a]["std_reward"]   for a in algos]
    sr     = [results[a]["success_rate"] * 100 for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Final Performance Comparison (100 eval episodes)",
                 fontsize=13, fontweight="bold")

    # Mean reward
    bars = axes[0].bar(labels, mean_r, yerr=std_r, color=colors,
                       capsize=6, edgecolor="black", linewidth=0.8)
    axes[0].set_ylabel("Mean Episode Reward")
    axes[0].set_title("Mean Episode Reward ± Std")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mean_r):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    # Success rate
    bars2 = axes[1].bar(labels, sr, color=colors,
                        edgecolor="black", linewidth=0.8)
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_title("Parking Success Rate")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, sr):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "comparison_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_reward_distribution(results: dict, save_dir: str):
    """Box plot of reward distributions per algorithm."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Episode Reward Distribution (100 eval episodes)",
                 fontsize=13, fontweight="bold")

    algos  = list(results.keys())
    labels = [LABELS[a] for a in algos]
    data   = [results[a]["rewards"] for a in algos]
    colors = [COLORS[a] for a in algos]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Episode Reward")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "reward_distribution.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       type=str, default=None,
                        choices=["dqn", "a2c", "ppo"])
    parser.add_argument("--ckpt",       type=str, default=None,
                        help="Path to checkpoint file for single-algo eval")
    parser.add_argument("--compare",    action="store_true",
                        help="Compare all algorithms from logs")
    parser.add_argument("--log_dir",    type=str, default="logs")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--noise_std",  type=float, default=0.02)
    parser.add_argument("--device",     type=str, default="cpu")
    return parser.parse_args()


def load_agent(algo: str, ckpt_path: str, obs_dim: int, env, device: str):
    if algo == "dqn":
        agent = DQNAgent(obs_dim=obs_dim, n_actions=env.N_DISCRETE_ACTIONS, device=device)
    elif algo == "a2c":
        agent = A2CAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    elif algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    agent.load(ckpt_path)
    return agent


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- Single algorithm evaluation ----
    if args.algo and args.ckpt:
        discrete = (args.algo == "dqn")
        env = ParkingEnv(discrete=discrete, noise_std=args.noise_std)
        obs_dim = env.observation_space.shape[0]

        agent = load_agent(args.algo, args.ckpt, obs_dim, env, args.device)
        print(f"\nEvaluating {args.algo.upper()} for {args.n_episodes} episodes...")
        result = evaluate_agent(env, agent, args.algo, args.n_episodes)

        print(f"\n{'='*40}")
        print(f"  Mean Reward  : {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Success Rate : {result['success_rate']:.2%}")
        print(f"  Mean Length  : {result['mean_length']:.1f} steps")
        print(f"{'='*40}")
        env.close()

    # ---- Compare all algorithms ----
    if args.compare:
        results = {}
        ckpt_dir = os.path.join(args.log_dir, "checkpoints")

        for algo in ["dqn", "a2c", "ppo"]:
            ckpt_path = os.path.join(ckpt_dir, f"{algo}_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] {algo}: checkpoint not found at {ckpt_path}")
                continue

            discrete = (algo == "dqn")
            env = ParkingEnv(discrete=discrete, noise_std=args.noise_std)
            obs_dim = env.observation_space.shape[0]

            agent = load_agent(algo, ckpt_path, obs_dim, env, args.device)
            print(f"Evaluating {algo.upper()}...")
            results[algo] = evaluate_agent(env, agent, algo, args.n_episodes)
            env.close()

            print(f"  -> reward={results[algo]['mean_reward']:.2f}  "
                  f"success={results[algo]['success_rate']:.2%}")

        if results:
            print("\nGenerating plots...")
            plot_training_curves(args.log_dir, args.log_dir)
            plot_comparison_bar(results, args.log_dir)
            plot_reward_distribution(results, args.log_dir)
            print("\nAll plots saved to:", args.log_dir)
        else:
            print("No checkpoints found. Train first with train.py")


if __name__ == "__main__":
    main()
