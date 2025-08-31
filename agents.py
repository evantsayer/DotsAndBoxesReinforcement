# agents.py
from __future__ import annotations
import os
import numpy as np
import torch

from dots_and_boxes import DotsAndBoxesEnv
from model import DQN

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

class BaseAgent:
    def select_action(self, env: DotsAndBoxesEnv) -> int:
        raise NotImplementedError

class RandomAgent(BaseAgent):
    def select_action(self, env: DotsAndBoxesEnv) -> int:
        legal = np.flatnonzero(env.get_legal_actions())
        return int(np.random.choice(legal)) if len(legal) > 0 else 0

class GreedyAgent(BaseAgent):
    """Chooses the move that immediately completes the most boxes; ties random."""
    def select_action(self, env: DotsAndBoxesEnv) -> int:
        legal_idxs = np.flatnonzero(env.get_legal_actions())
        if len(legal_idxs) == 0:
            return 0
        gains = []
        for a in legal_idxs:
            sim = env.clone()
            _, _, _, info = sim.step(a)
            gains.append(info.get("new_boxes", 0))
        gains = np.array(gains)
        best = np.flatnonzero(gains == gains.max())
        return int(legal_idxs[np.random.choice(best)])

class DQNAgent(BaseAgent):
    def __init__(self, model_path: str, device: str = None):
        self.device = device or pick_device()
        ckpt = torch.load(model_path, map_location=self.device)
        input_dim = ckpt.get("input_dim")
        action_dim = ckpt.get("action_dim")
        self.net = DQN(input_dim, action_dim).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

    def select_action(self, env: DotsAndBoxesEnv) -> int:
        state = torch.from_numpy(env.get_state_vector()).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.net(state).squeeze(0).cpu().numpy()
        mask = env.get_legal_actions()
        q[~mask] = -1e9
        best = np.flatnonzero(q == q.max())
        if len(best) == 0:
            legal = np.flatnonzero(mask)
            return int(np.random.choice(legal)) if len(legal) else 0
        return int(np.random.choice(best))

def build_opponent(name: str) -> BaseAgent:
    name = (name or "").strip().lower()
    if name == "greedy":
        return GreedyAgent()
    elif name.startswith("file:"):
        path = name.split("file:", 1)[1]
        return DQNAgent(path)
    else:
        return RandomAgent()

def list_model_files(models_dir: str = "models"):
    if not os.path.isdir(models_dir):
        return []
    return [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith(".pth")]
