# plotting.py
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.copy()
    w = min(w, len(x))
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w)/w, mode="valid")

def _to_win_numeric(df: pd.DataFrame) -> np.ndarray:
    if "win" in df.columns:
        win = df["win"].to_numpy(dtype=float)
        tie = df["tie"].to_numpy(dtype=float) if "tie" in df.columns else np.zeros_like(win)
        return win + 0.5 * tie
    elif "score_diff" in df.columns:
        sd = df["score_diff"].to_numpy(dtype=float)
        return (sd > 0).astype(float)
    else:
        return np.zeros(len(df), dtype=float)

def save_training_plots(df: pd.DataFrame, out_dir: str, title: Optional[str] = None):
    """Save a suite of polished training plots for a single run."""
    _ensure_dir(out_dir)
    model_name = str(df.get("model_name", ["model"])[0])
    run_id = str(df.get("run_id", ["run"])[0])

    # Common series
    ep = df["episode"].to_numpy()
    rew = df["reward"].to_numpy(dtype=float)
    mean100 = df["mean100"].to_numpy(dtype=float)
    eps = df["epsilon"].to_numpy(dtype=float)
    win_num = _to_win_numeric(df)
    cum_wins = np.cumsum((df["win"].to_numpy(dtype=float) if "win" in df.columns else (win_num == 1).astype(float)))

    # Rolling windows
    def _moving_avg_local(x: np.ndarray):
        w = max(10, min(100, len(x)//10))
        return _moving_avg(x, w), w

    win_rate, win_window = _moving_avg_local(win_num)

    # Plot 1: Reward per episode (with rolling mean)
    plt.figure(figsize=(12, 6))
    plt.plot(ep, rew, linewidth=1, alpha=0.35, label="Episode reward")
    plt.plot(ep[-len(mean100):], mean100, linewidth=2, label="Rolling mean (100)")
    plt.title(title or f"{model_name} — Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_curve.png"), dpi=160)
    plt.close()

    # Plot 2: Rolling win rate
    plt.figure(figsize=(12, 6))
    if len(win_rate) > 0:
        plt.plot(ep[-len(win_rate):], win_rate, linewidth=2, label=f"Rolling win score (w={win_window})")
    plt.ylim(0, 1.05)
    plt.title(f"{model_name} — Rolling Win Score")
    plt.xlabel("Episode")
    plt.ylabel("Win score (win=1, tie=0.5, loss=0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rolling_win_rate.png"), dpi=160)
    plt.close()

    # Plot 3: Epsilon schedule
    plt.figure(figsize=(12, 6))
    plt.plot(ep, eps, linewidth=2)
    plt.title(f"{model_name} — Epsilon Schedule")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epsilon_schedule.png"), dpi=160)
    plt.close()

    # Plot 4: Score difference
    if "score_diff" in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(ep, df["score_diff"].to_numpy(dtype=float), linewidth=2)
        plt.axhline(0, color="k", linewidth=1)
        plt.title(f"{model_name} — Score Difference (Agent − Best Opponent)")
        plt.xlabel("Episode")
        plt.ylabel("Score diff")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "score_diff.png"), dpi=160)
        plt.close()

    # Plot 5: Cumulative wins
    plt.figure(figsize=(12, 6))
    plt.plot(ep, cum_wins, linewidth=2)
    plt.title(f"{model_name} — Cumulative Wins")
    plt.xlabel("Episode")
    plt.ylabel("Wins (cumulative)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cumulative_wins.png"), dpi=160)
    plt.close()

    # Plot 6: Opponent schedule (per-episode)
    if "opponent_for_episode" in df.columns:
        mapping = {"random": 0, "greedy": 1}
        opp_series = [mapping.get(str(x).lower(), np.nan) for x in df["opponent_for_episode"].tolist()]
        plt.figure(figsize=(12, 3.5))
        plt.step(ep, opp_series, where="post", linewidth=2)
        plt.yticks([0, 1], ["random", "greedy"])
        plt.ylim(-0.5, 1.5)
        plt.title(f"{model_name} — Opponent per Episode")
        plt.xlabel("Episode")
        plt.grid(True, axis="x", alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "opponent_per_episode.png"), dpi=160)
        plt.close()

    # Plot 7: Open-3 traps per episode
    if "open3_traps" in df.columns:
        traps = df["open3_traps"].to_numpy(dtype=float)
        plt.figure(figsize=(12, 4))
        plt.bar(ep, traps, width=1.0)
        plt.title(f"{model_name} — Open-3 Traps Left per Episode")
        plt.xlabel("Episode")
        plt.ylabel("# of open-3 boxes left")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "open3_traps.png"), dpi=160)
        plt.close()
