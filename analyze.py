# analyze.py
import os
import glob
from typing import Dict, List

# Force non-interactive backend (safe for threads & headless)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

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

def _short_label(df: pd.DataFrame) -> str:
    model = str(df.get("model_name", ["model"])[0])
    opp = str(df.get("opponent_schedule", ["?"])[0])
    lr = float(df.get("lr", [0.0])[0])
    rid = str(df.get("run_id", [""])[0])[-4:]
    return f"{model}|{opp}|lr={lr:.0e}|{rid}"

def _final_stats(df: pd.DataFrame) -> Dict:
    win_num = _to_win_numeric(df)
    n = len(win_num)
    tail = max(10, n // 10)
    final_winrate = float(np.mean(win_num[-tail:])) if n else 0.0
    best_mean100 = float(df["mean100"].iloc[-1]) if "mean100" in df.columns and len(df) else 0.0
    final_score_diff = float(np.mean(df["score_diff"].to_numpy(dtype=float)[-tail:])) if "score_diff" in df.columns and n else 0.0
    return {
        "final_winrate": final_winrate,
        "best_mean100": best_mean100,
        "final_score_diff": final_score_diff,
    }

def _read_all_csv(csv_dir="csv") -> List[pd.DataFrame]:
    paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__csv_path__"] = p
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return dfs

def run_aggregate(csv_dir: str = "csv", out_dir: str = "plots/aggregate", top_k: int = 8) -> Dict:
    _ensure_dir(out_dir)
    dfs = _read_all_csv(csv_dir)
    if not dfs:
        print("No CSVs found in 'csv/'. Train a model first.")
        return {"ok": False, "out_dir": out_dir, "files": []}

    summaries = []
    for df in dfs:
        stats = _final_stats(df)
        row = {
            "model_name": str(df.get("model_name", ["model"])[0]),
            "run_id": str(df.get("run_id", ["run"])[0]),
            "episodes_total": int(df.get("episodes_total", [len(df)])[0]),
            "opponent_schedule": str(df.get("opponent_schedule", ["?"])[0]),
            "curriculum_switch_ep": int(df.get("curriculum_switch_ep", [0])[0]),
            "lr": float(df.get("lr", [0.0])[0]),
            "gamma": float(df.get("gamma", [0.99])[0]),
            "open_box_penalty": float(df.get("open_box_penalty", [0.0])[0]),
            "combo_multiplier": float(df.get("combo_multiplier", [1.0])[0]),
            "players": int(df.get("players", [2])[0]),
            **stats,
            "csv_path": df["__csv_path__"].iloc[0],
        }
        summaries.append(row)
    summary_df = pd.DataFrame(summaries)
    summary_csv = os.path.join("csv", "aggregate_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Plot: LR vs Final Winrate
    plt.figure(figsize=(9, 6))
    schedules = summary_df["opponent_schedule"].unique().tolist()
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for i, sch in enumerate(schedules):
        sub = summary_df[summary_df["opponent_schedule"] == sch]
        plt.scatter(sub["lr"], sub["final_winrate"], s=64, marker=markers[i % len(markers)], label=sch)
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Final win score (last 10%)")
    plt.title("Learning Rate vs Final Win Score")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Opponent schedule")
    plt.tight_layout()
    lr_plot = os.path.join(out_dir, "lr_vs_final_winrate.png")
    plt.savefig(lr_plot, dpi=160)
    plt.close()

    # Plot: Episodes vs Final Winrate
    plt.figure(figsize=(9, 6))
    for i, sch in enumerate(schedules):
        sub = summary_df[summary_df["opponent_schedule"] == sch]
        plt.scatter(sub["episodes_total"], sub["final_winrate"], s=64, marker=markers[i % len(markers)], label=sch)
    plt.xlabel("# Episodes")
    plt.ylabel("Final win score (last 10%)")
    plt.title("Episodes vs Final Win Score")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Opponent schedule")
    plt.tight_layout()
    ep_plot = os.path.join(out_dir, "episodes_vs_final_winrate.png")
    plt.savefig(ep_plot, dpi=160)
    plt.close()

    # Plot: Open-box penalty vs Final Winrate
    plt.figure(figsize=(9, 6))
    for i, sch in enumerate(schedules):
        sub = summary_df[summary_df["opponent_schedule"] == sch]
        plt.scatter(sub["open_box_penalty"], sub["final_winrate"], s=64, marker=markers[i % len(markers)], label=sch)
    plt.xlabel("Open-box penalty")
    plt.ylabel("Final win score (last 10%)")
    plt.title("Penalty Strength vs Final Win Score")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Opponent schedule")
    plt.tight_layout()
    pen_plot = os.path.join(out_dir, "penalty_vs_final_winrate.png")
    plt.savefig(pen_plot, dpi=160)
    plt.close()

    # Plot: Final Winrate by Opponent Schedule (bar)
    plt.figure(figsize=(9, 6))
    bars = summary_df.groupby("opponent_schedule")["final_winrate"].mean().sort_values()
    plt.bar(bars.index, bars.values)
    plt.ylabel("Mean final win score")
    plt.title("Final Win Score by Opponent Schedule")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    sch_plot = os.path.join(out_dir, "final_winrate_by_schedule.png")
    plt.savefig(sch_plot, dpi=160)
    plt.close()

    # Plot: Overlay smoothed reward curves for top runs
    top_k = 8
    top = summary_df.sort_values("final_winrate", ascending=False).head(top_k)
    plt.figure(figsize=(12, 7))
    for _, row in top.iterrows():
        df = pd.read_csv(row["csv_path"])
        y = df["reward"].to_numpy(dtype=float)
        ep = df["episode"].to_numpy()
        w = max(10, min(100, len(y)//10))
        y_sm = _moving_avg(y, w)
        plt.plot(ep[-len(y_sm):], y_sm, linewidth=2, label=_short_label(df))
    plt.title("Smoothed Reward Curves — Top Runs")
    plt.xlabel("Episode")
    plt.ylabel("Reward (smoothed)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=1, fontsize=9)
    plt.tight_layout()
    overlay_plot = os.path.join(out_dir, "overlay_reward_curves_top.png")
    plt.savefig(overlay_plot, dpi=160)
    plt.close()

    # Optional: simple correlations
    corr_df = summary_df[["final_winrate", "lr", "episodes_total", "open_box_penalty", "combo_multiplier"]].copy()
    corr = corr_df.corr(numeric_only=True)
    corr_csv = os.path.join(out_dir, "aggregate_correlations.csv")
    corr.to_csv(corr_csv)

    return {"ok": True, "out_dir": out_dir,
            "files": [lr_plot, ep_plot, pen_plot, sch_plot, overlay_plot, summary_csv, corr_csv]}

if __name__ == "__main__":
    out = run_aggregate()
    print("Aggregate analysis:", out)
