# train.py
from __future__ import annotations
import os
import argparse
from typing import Callable, Optional, List
from datetime import datetime

# Ensure a safe matplotlib backend if anything imports pyplot before plotting.py
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import TrainConfig
from dots_and_boxes import DotsAndBoxesEnv
from model import DQN
from agents import RandomAgent, GreedyAgent
from plotting import save_training_plots

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity,), dtype=np.int64)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, s, a, r, ns, d):
        i = self.ptr % self.capacity
        self.state[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.next_state[i] = ns
        self.done[i] = d
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        s = torch.from_numpy(self.state[idx])
        a = torch.from_numpy(self.action[idx])
        r = torch.from_numpy(self.reward[idx])
        ns = torch.from_numpy(self.next_state[idx])
        d = torch.from_numpy(self.done[idx])
        return s, a, r, ns, d

# Helpers
def epsilon_by_episode(ep, cfg: TrainConfig):
    if ep >= cfg.eps_decay_episodes:
        return cfg.eps_end
    frac = 1 - (ep / max(1, cfg.eps_decay_episodes))
    return cfg.eps_end + (cfg.eps_start - cfg.eps_end) * frac

def make_opponents(kind: str, count: int) -> List:
    if kind == "greedy":
        return [GreedyAgent() for _ in range(count)]
    return [RandomAgent() for _ in range(count)]

def pick_device() -> str:
    # Prefer Apple MPS on macOS, else CUDA, else CPU
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def optimize_dqn(online: DQN, target: DQN, buffer: ReplayBuffer, optimizer, cfg: TrainConfig, device):
    if buffer.size < cfg.batch_size:
        return 0.0
    criterion = nn.SmoothL1Loss()
    s, a, r, ns, d = buffer.sample(cfg.batch_size)
    s = s.to(device)
    ns = ns.to(device)
    a = a.to(device)
    r = r.to(device)
    d = d.to(device)

    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q = target(ns).max(1)[0]
        target_q = r + (1 - d) * cfg.gamma * next_q
    loss = criterion(q, target_q)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())

def opponent_kind_for_episode(opponent_mode: str, ep: int, cfg: TrainConfig) -> str:
    mode = (opponent_mode or "random").lower()
    if mode == "alternate":
        # even episodes random, odd episodes greedy
        return "random" if (ep % 2 == 0) else "greedy"
    if mode == "curriculum":
        # start easy, switch to greedy at curriculum_switch_ep
        return "random" if ep < cfg.curriculum_switch_ep else "greedy"
    # fixed
    return "greedy" if mode == "greedy" else "random"

# Training Loop (penalize ALL open freebies at turn end)
def train_dqn(cfg: TrainConfig, progress_cb: Optional[Callable[[dict], None]] = None):
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    os.makedirs("csv", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    device = pick_device()

    env = DotsAndBoxesEnv(
        n_players=cfg.players,
        base_box_reward=cfg.base_box_reward,
        combo_multiplier=cfg.combo_multiplier,
        start_player_random=cfg.start_player_random,
        seed=cfg.seed,
    )
    state_dim = len(env.get_state_vector())
    action_dim = env.action_space

    online = DQN(state_dim, action_dim).to(device)
    target = DQN(state_dim, action_dim).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=cfg.lr)

    buffer = ReplayBuffer(cfg.buffer_size, state_dim)

    total_steps = 0
    best_mean = -1e9
    reward_history: List[float] = []

    # Log Metrics
    ep_idx, ep_reward_list, mean100_list, eps_list, opp_used_list = [], [], [], [], []
    win_list, tie_list, agent_score_list, best_score_list, score_diff_list = [], [], [], [], []
    open3_traps_list = []

    model_base = os.path.splitext(os.path.basename(cfg.save_path))[0]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    plots_dir = os.path.join("plots", model_base, run_id)
    os.makedirs(plots_dir, exist_ok=True)

    for ep in range(cfg.episodes):
        opp_kind = opponent_kind_for_episode(cfg.opponent, ep, cfg)
        opponents = make_opponents(opp_kind, cfg.players - 1)

        state = env.reset()

        while env.current_player != 0 and not env.is_done():
            a = opponents[(env.current_player - 1) % len(opponents)].select_action(env)
            env.step(a)

        ep_reward = 0.0
        done = env.is_done()
        open3_sum_episode = 0

        eps = epsilon_by_episode(ep, cfg)

        while not done:
            eps = epsilon_by_episode(ep, cfg)
            mask = env.get_legal_actions()
            if np.random.rand() < eps:
                legal = np.flatnonzero(mask)
                action = int(np.random.choice(legal)) if len(legal) else 0
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(env.get_state_vector()).float().unsqueeze(0).to(device)
                    q = online(state_t).squeeze(0).cpu().numpy()
                q[~mask] = -1e9
                best = np.flatnonzero(q == q.max())
                action = int(np.random.choice(best)) if len(best) else 0

            _, r_agent, done, info = env.step(action)

            penalty = 0.0
            if info.get("turn_passed", False):
                open3_total = int(info.get("open3_total", 0))
                if open3_total > 0:
                    penalty = cfg.open_box_penalty * float(open3_total)
                    open3_sum_episode += open3_total

            while not done and env.current_player != 0:
                oa = opponents[(env.current_player - 1) % len(opponents)].select_action(env)
                _, _, done, _ = env.step(oa)

            next_state_final = env.get_state_vector()
            total_r = float(r_agent) - penalty

            buffer.push(state, action, total_r, next_state_final, float(done))
            state = next_state_final
            ep_reward += total_r

            _ = optimize_dqn(online, target, buffer, optimizer, cfg, device)
            total_steps += 1
            if total_steps % cfg.target_update == 0:
                target.load_state_dict(online.state_dict())

        # Compute Scores
        scores = env.scores.copy()
        a_score = int(scores[0])
        best_score = int(scores.max())
        is_best = scores == best_score
        winners = int(is_best.sum())
        win = 1 if (is_best[0] and winners == 1) else 0
        tie = 1 if (is_best[0] and winners > 1) else 0
        score_diff = a_score - int(scores[1:].max()) if len(scores) > 1 else a_score

        reward_history.append(ep_reward)
        mean_100 = float(np.mean(reward_history[-100:]))

        # Save best model snapshot
        if mean_100 > best_mean:
            best_mean = mean_100
            torch.save({
                "state_dict": online.state_dict(),
                "input_dim": state_dim,
                "action_dim": action_dim,
                "config": vars(cfg),
                "episode": ep,
                "mean_100": mean_100,
                "device": device,
            }, cfg.save_path)

        # Append per-episode logs
        ep_idx.append(ep + 1)
        ep_reward_list.append(float(ep_reward))
        mean100_list.append(mean_100)
        eps_list.append(float(eps))
        opp_used_list.append(opp_kind)
        win_list.append(win)
        tie_list.append(tie)
        agent_score_list.append(a_score)
        best_score_list.append(best_score)
        score_diff_list.append(int(score_diff))
        open3_traps_list.append(int(open3_sum_episode))

        if progress_cb and (ep % 10 == 0 or ep == cfg.episodes - 1):
            progress_cb({
                "episode": ep + 1,
                "episodes": cfg.episodes,
                "ep_reward": float(ep_reward),
                "mean_100": mean_100,
                "epsilon": float(eps),
                "saved": cfg.save_path,
                "device": device,
                "opponents": opp_kind,
            })

    # final save
    torch.save({
        "state_dict": online.state_dict(),
        "input_dim": state_dim,
        "action_dim": action_dim,
        "config": vars(cfg),
        "episode": cfg.episodes,
        "mean_100": float(np.mean(reward_history[-100:])),
        "device": device,
    }, cfg.save_path)

    # Save metrics to CSV
    df = pd.DataFrame({
        "episode": ep_idx,
        "reward": ep_reward_list,
        "mean100": mean100_list,
        "epsilon": eps_list,
        "opponent_for_episode": opp_used_list,
        "win": win_list,
        "tie": tie_list,
        "agent_score": agent_score_list,
        "best_score": best_score_list,
        "score_diff": score_diff_list,
        "open3_traps": open3_traps_list,
    })
    # Add metadata columns (same on every row, for aggregation)
    df["model_name"] = model_base
    df["run_id"] = run_id
    df["save_path"] = cfg.save_path
    df["device"] = device
    df["episodes_total"] = cfg.episodes
    df["players"] = cfg.players
    df["opponent_schedule"] = cfg.opponent
    df["curriculum_switch_ep"] = cfg.curriculum_switch_ep
    df["lr"] = cfg.lr
    df["gamma"] = cfg.gamma
    df["batch_size"] = cfg.batch_size
    df["buffer_size"] = cfg.buffer_size
    df["target_update"] = cfg.target_update
    df["eps_start"] = cfg.eps_start
    df["eps_end"] = cfg.eps_end
    df["eps_decay_episodes"] = cfg.eps_decay_episodes
    df["open_box_penalty"] = cfg.open_box_penalty
    df["combo_multiplier"] = cfg.combo_multiplier
    df["base_box_reward"] = cfg.base_box_reward
    df["start_player_random"] = cfg.start_player_random
    df["timestamp"] = datetime.now().isoformat(timespec="seconds")

    csv_name = f"{model_base}__{run_id}.csv"
    csv_path = os.path.join("csv", csv_name)
    df.to_csv(csv_path, index=False)

    # Generate polished plots
    try:
        save_training_plots(
            df=df,
            out_dir=plots_dir,
            title=f"{model_base} (run {run_id}) — device: {device}, opp: {cfg.opponent}"
        )
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    return cfg.save_path

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Train DQN on Dots and Boxes")
    p.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--eps_start", type=float, default=TrainConfig.eps_start)
    p.add_argument("--eps_end", type=float, default=TrainConfig.eps_end)
    p.add_argument("--eps_decay_episodes", type=int, default=TrainConfig.eps_decay_episodes)
    p.add_argument("--target_update", type=int, default=TrainConfig.target_update)
    p.add_argument("--buffer_size", type=int, default=TrainConfig.buffer_size)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--players", type=int, default=TrainConfig.players, choices=[2,3,4])
    p.add_argument("--opponent", type=str, default=TrainConfig.opponent,
                   choices=["random","greedy","alternate","curriculum"])
    p.add_argument("--curriculum_switch_ep", type=int, default=TrainConfig.curriculum_switch_ep)
    p.add_argument("--combo_multiplier", type=float, default=TrainConfig.combo_multiplier)
    p.add_argument("--base_box_reward", type=float, default=TrainConfig.base_box_reward)
    p.add_argument("--open_box_penalty", type=float, default=TrainConfig.open_box_penalty)
    p.add_argument("--save_path", type=str, default=TrainConfig.save_path)
    p.add_argument("--start_player_random", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        seed=args.seed,
        players=args.players,
        opponent=args.opponent,
        curriculum_switch_ep=args.curriculum_switch_ep,
        combo_multiplier=args.combo_multiplier,
        base_box_reward=args.base_box_reward,
        open_box_penalty=args.open_box_penalty,
        save_path=args.save_path,
        start_player_random=args.start_player_random,
    )
    def cb(info):
        print(f"[{info['episode']}/{info['episodes']}] device={info['device']} "
              f"opp={info['opponents']} reward={info['ep_reward']:.2f} "
              f"mean100={info['mean_100']:.2f} eps={info['epsilon']:.3f}")
    path = train_dqn(cfg, progress_cb=cb)
    print("Saved model to:", path)
