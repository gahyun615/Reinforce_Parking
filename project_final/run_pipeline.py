"""
run_pipeline.py
- Train all configured algorithms sequentially
- Run final comparison evaluation automatically
- Record demo videos for each trained algorithm

Usage:
    python run_pipeline.py
    python run_pipeline.py --total_steps 200000 --n_episodes 100
    python run_pipeline.py --skip_train --skip_record
"""

import argparse
import os
import subprocess
import sys
import importlib.util

import torch


ALGORITHMS = ["dqn", "double_dqn", "reinforce", "a2c", "ppo", "sac"]


def resolve_device(requested: str) -> str:
    """Use CUDA when requested and available; fail fast otherwise."""
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but no GPU is available. "
                "Install a CUDA-enabled PyTorch build or pass --device cpu."
            )
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return requested
    return requested


def run_cmd(cmd: list[str]):
    print("\n" + "=" * 80)
    print("RUN:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def print_video_backend_status():
    """
    Print whether mp4 backend dependency is available.
    record.py will fall back to gif when mp4 backend is unavailable.
    """
    has_ffmpeg_backend = importlib.util.find_spec("imageio_ffmpeg") is not None
    if has_ffmpeg_backend:
        print("[VIDEO] imageio-ffmpeg detected: mp4 recording is available.")
    else:
        print(
            "[VIDEO] imageio-ffmpeg not found: record.py will save gif fallback "
            "if mp4 writer is unavailable."
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=300_000)
    parser.add_argument("--noise_std", type=float, default=0.02)
    parser.add_argument("--n_vehicles", type=int, default=6)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--save_every", type=int, default=50_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--record_episodes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="videos")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training and only run compare evaluation.",
    )
    parser.add_argument(
        "--skip_record",
        action="store_true",
        help="Skip video recording after evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    print_video_backend_status()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_dir = os.path.join(args.log_dir, "checkpoints")

    if not args.skip_train:
        for algo in ALGORITHMS:
            run_cmd([
                sys.executable,
                "train.py",
                "--algo", algo,
                "--total_steps", str(args.total_steps),
                "--noise_std", str(args.noise_std),
                "--n_vehicles", str(args.n_vehicles),
                "--log_dir", args.log_dir,
                "--save_every", str(args.save_every),
                "--device", args.device,
            ])

    run_cmd([
        sys.executable,
        "evaluate.py",
        "--compare",
        "--log_dir", args.log_dir,
        "--n_episodes", str(args.n_episodes),
        "--noise_std", str(args.noise_std),
        "--device", args.device,
    ])

    if not args.skip_record:
        for algo in ALGORITHMS:
            ckpt_path = os.path.join(ckpt_dir, f"{algo}_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] record {algo}: checkpoint not found at {ckpt_path}")
                continue

            try:
                run_cmd([
                    sys.executable,
                    "record.py",
                    "--algo", algo,
                    "--ckpt", ckpt_path,
                    "--n_episodes", str(args.record_episodes),
                    "--output_dir", args.output_dir,
                    "--fps", str(args.fps),
                    "--noise_std", str(args.noise_std),
                    "--device", args.device,
                ])
            except subprocess.CalledProcessError:
                print(f"[SKIP] record {algo}: recording failed (see logs above).")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
