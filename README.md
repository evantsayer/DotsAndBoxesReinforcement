# Dots & Boxes Reinforcement Learning

## Overview

A compact reinforcement learning (DQN) project that learns to play the **Dots & Boxes** board game on a fixed 6×6 grid. The app includes:

* A **pygame** user interface for playing (human vs AI or AI vs AI), training, and running aggregate analysis
* A **DQN** training loop with reward shaping and opponent scheduling
* Automated **plotting** and **CSV logging** for each run

## Supported Platforms

Developed on **macOS** (Apple Silicon) with PyTorch MPS acceleration. The code is cross-platform and should also work on **Linux** and **Windows**. If GPU acceleration isn’t available (CUDA on Windows/Linux, MPS on macOS), training falls back to **CPU** automatically.

## Requirements

* **Python 3.9+** (recommended)
* Packages:

  * `torch` (PyTorch with optional CUDA/MPS)
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `pygame`

## Quick install (use a virtual environment)

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pandas matplotlib pygame
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch numpy pandas matplotlib pygame
```

**(Optional) Verify acceleration**

```bash
python - <<'PY'
import torch
print("MPS available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())
PY
```

## Running the App (GUI)

```bash
python app.py
```

From the main menu you can:

* **Play**
  Set number of players (2–4). Choose for each seat: `Human`, `AI:Random`, `AI:Greedy`, or `AI:Model`.
  For `AI:Model`, pick a saved `.pth` file in `models/`. Press **F5** on the setup screen to refresh the model list after training.

* **Train**
  Configure: episodes, batch size, learning rate, discount $\gamma$, ε-greedy schedule, opponent schedule (`random` / `greedy` / `alternate` / `curriculum`), curriculum switch episode, reward shaping (base box reward, combo multiplier), open-box penalty, and model save path. A live status panel shows progress.

* **Analyze**
  Runs aggregate analysis across all CSV logs in `csv/`, producing summary plots and tables.

## Command-Line Training (optional)

```bash
python train.py \
  --episodes 10000 --lr 1e-3 --players 2 \
  --opponent curriculum --curriculum_switch_ep 9000 \
  --open_box_penalty 5 --combo_multiplier 1.5 \
  --save_path models/dqn_dots_and_boxes.pth
```

## Key Files & Folders

* `app.py` — Launches the pygame GUI (Play / Train / Analyze)
* `train.py` — DQN training loop + CLI
* `dots_and_boxes.py` — Game environment and rules
* `model.py` — Neural network (DQN)
* `agents.py` — Random, Greedy, and DQN agents
* `plotting.py` — Per-run plots
* `analyze.py` — Cross-run aggregation and plots
* `config.py` — UI settings, board size, default `TrainConfig`

## Outputs & Artifacts

On first run, the app creates these folders automatically:

* **`models/`** — Saved PyTorch checkpoints (`*.pth`).
  Default: `models/dqn_dots_and_boxes.pth`
* **`csv/`** — One CSV per training run (e.g., `<model_base>__<run_id>.csv`) with episode-by-episode metrics
* **`plots/`**

  * `<model_base>/<run_id>/` — Per-run figures:

    * `reward_curve.png`, `rolling_win_rate.png`, `epsilon_schedule.png`, `score_diff.png`*,
      `cumulative_wins.png`, `opponent_per_episode.png`*, `open3_traps.png` (\*if available)
  * `aggregate/` — Cross-run summaries:

    * `lr_vs_final_winrate.png`, `episodes_vs_final_winrate.png`, `penalty_vs_final_winrate.png`,
      `final_winrate_by_schedule.png`, `overlay_reward_curves_top.png`, `aggregate_correlations.csv`
* **`csv/aggregate_summary.csv`** — One-line summary per run (for quick comparisons)

## Troubleshooting
* Pygame window issues over SSH: ensure a local display or desktop session.
* For GPU use:

  * **macOS**: install a recent PyTorch build with **MPS**; the app auto-detects MPS.
  * **Windows/Linux**: install a **CUDA-enabled** PyTorch build compatible with your driver.

## Copyright

This project is for coursework and educational use. Feel free to use any of this code.
