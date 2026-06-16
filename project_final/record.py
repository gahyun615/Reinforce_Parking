"""
record.py
- Load a trained checkpoint and record episodes as mp4

Usage:
    python record.py --algo ppo --ckpt logs/checkpoints/ppo_final.pt
    python record.py --algo dqn --ckpt logs/checkpoints/dqn_final.pt --n_episodes 3
"""

import os
import argparse
import sys
import numpy as np
import imageio

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.parking_env import ParkingEnv
from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.reinforce import REINFORCEAgent
from agents.a2c import A2CAgent
from agents.ppo import PPOAgent
from agents.checkpoint_utils import CheckpointMismatchError


def save_frames(out_path: str, frames, fps: int) -> str:
    """
    Save frames as mp4 when ffmpeg is available.
    Falls back to gif when mp4 writer is unavailable.
    Returns the actual saved path.
    """
    try:
        # Force ffmpeg plugin so imageio does not choose a non-video writer.
        imageio.mimsave(out_path, frames, fps=fps, format="FFMPEG")
        return out_path
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        # GIF expects duration per frame (seconds), not fps.
        imageio.mimsave(gif_path, frames, duration=1.0 / max(fps, 1))
        print(f"  [WARN] mp4 save failed ({e}). Saved gif instead: {gif_path}")
        return gif_path


def load_agent(algo, ckpt_path, obs_dim, env, device):
    if algo == "dqn":
        agent = DQNAgent(obs_dim=obs_dim, n_actions=env.N_DISCRETE_ACTIONS, device=device)
    elif algo == "double_dqn":
        agent = DoubleDQNAgent(obs_dim=obs_dim, n_actions=env.N_DISCRETE_ACTIONS, device=device)
    elif algo == "reinforce":
        agent = REINFORCEAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    elif algo == "a2c":
        agent = A2CAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    elif algo == "ppo":
        agent = PPOAgent(obs_dim=obs_dim, action_dim=env.action_space.shape[0], device=device)
    agent.load(ckpt_path)
    return agent


def record_episodes(algo, ckpt_path, n_episodes, output_dir, device, noise_std, fps):
    os.makedirs(output_dir, exist_ok=True)

    discrete = (algo in ["dqn", "double_dqn"])
    env = ParkingEnv(discrete=discrete, noise_std=noise_std, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]

    try:
        agent = load_agent(algo, ckpt_path, obs_dim, env, device)
    except CheckpointMismatchError as e:
        print(f"[ERROR] {e}")
        env.close()
        sys.exit(1)
    print(f"Loaded {algo.upper()} from {ckpt_path}")

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        frames = []
        done = False
        ep_reward = 0.0

        while not done:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if algo in ["dqn", "double_dqn"]:
                action = agent.select_action(obs, training=False)
            else:
                action, _, _ = agent.select_action(obs, training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

        success = env.is_success(obs, info)
        status = "SUCCESS" if success else "FAIL"

        out_path = os.path.join(output_dir, f"{algo}_ep{ep+1}_{status}.mp4")
        if frames:
            saved_path = save_frames(out_path, frames, fps)
            print(f"  Episode {ep+1}: reward={ep_reward:.1f}  {status}  -> {saved_path}")
        else:
            print(f"  Episode {ep+1}: no frames captured")

    env.close()
    print(f"\nDone. Videos saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",       type=str, required=True,
                        choices=["dqn", "double_dqn", "reinforce", "a2c", "ppo"])
    parser.add_argument("--ckpt",       type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="videos")
    parser.add_argument("--fps",        type=int, default=30)
    parser.add_argument("--noise_std",  type=float, default=0.02)
    parser.add_argument("--device",     type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    record_episodes(
        algo=args.algo,
        ckpt_path=args.ckpt,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir,
        device=args.device,
        noise_std=args.noise_std,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
