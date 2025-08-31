# dots_and_boxes.py
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, Dict

from config import BOARD_ROWS, BOARD_COLS, EDGE_COUNT, BASE_BOX_REWARD, COMBO_MULTIPLIER

class DotsAndBoxesEnv:
    """
    Fixed 6x6 dots board (5x5 boxes).
    Action space indexes all edges: first horizontal (row-major), then vertical (row-major).

    Tracks edge owners so GUI can render player-colored lines.
    Provides step() info:
      - new_boxes: int
      - open3_adjacent: number of adjacent boxes to the LAST placed edge that have exactly 3 sides filled
      - open3_total: total number of 3-sided boxes on the WHOLE board AFTER this move
      - last_edge: (typ, r, c)
      - turn_passed: True if the turn switched after this action
    """
    def __init__(
        self,
        rows: int = BOARD_ROWS,
        cols: int = BOARD_COLS,
        n_players: int = 2,
        base_box_reward: float = BASE_BOX_REWARD,
        combo_multiplier: float = COMBO_MULTIPLIER,
        start_player_random: bool = True,
        seed: Optional[int] = None,
    ):
        assert 2 <= n_players <= 4
        self.rows = rows
        self.cols = cols
        self.n_players = n_players
        self.base_box_reward = float(base_box_reward)
        self.combo_multiplier = float(combo_multiplier)
        self.start_player_random = start_player_random
        self.rng = np.random.default_rng(seed)

        self.h_total = self.rows * (self.cols - 1)
        self.v_total = (self.rows - 1) * self.cols
        self.action_space = self.h_total + self.v_total  # 60
        assert self.action_space == EDGE_COUNT

        self.reset()

    # Index helpers
    def action_to_edge(self, a: int) -> Tuple[str, int, int]:
        """Return ('h' or 'v', r, c)."""
        if a < self.h_total:
            r = a // (self.cols - 1)
            c = a % (self.cols - 1)
            return ('h', r, c)
        else:
            a2 = a - self.h_total
            r = a2 // self.cols
            c = a2 % self.cols
            return ('v', r, c)

    def edge_to_action(self, typ: str, r: int, c: int) -> int:
        if typ == 'h':
            return r * (self.cols - 1) + c
        else:
            return self.h_total + r * self.cols + c

    # Core API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # 0/1 taken
        self.h_edges = np.zeros((self.rows, self.cols - 1), dtype=np.int8)
        self.v_edges = np.zeros((self.rows - 1, self.cols), dtype=np.int8)
        # owner of each edge (-1 = none)
        self.h_owner = -np.ones((self.rows, self.cols - 1), dtype=np.int8)
        self.v_owner = -np.ones((self.rows - 1, self.cols), dtype=np.int8)
        # owner of boxes (-1 = none)
        self.box_owner = -np.ones((self.rows - 1, self.cols - 1), dtype=np.int8)
        # scores
        self.scores = np.zeros(self.n_players, dtype=np.int16)
        # current player
        self.current_player = self.rng.integers(0, self.n_players) if self.start_player_random else 0
        # combo count within current player's turn
        self._turn_combo_count = 0
        # cached legal mask
        self._legal_mask = None
        return self.get_state_vector()

    def clone(self) -> "DotsAndBoxesEnv":
        env = DotsAndBoxesEnv(
            rows=self.rows,
            cols=self.cols,
            n_players=self.n_players,
            base_box_reward=self.base_box_reward,
            combo_multiplier=self.combo_multiplier,
            start_player_random=False,
        )
        env.h_edges = self.h_edges.copy()
        env.v_edges = self.v_edges.copy()
        env.h_owner = self.h_owner.copy()
        env.v_owner = self.v_owner.copy()
        env.box_owner = self.box_owner.copy()
        env.scores = self.scores.copy()
        env.current_player = self.current_player
        env._turn_combo_count = self._turn_combo_count
        env._legal_mask = None if self._legal_mask is None else self._legal_mask.copy()
        return env

    def is_done(self) -> bool:
        return (self.box_owner != -1).sum() == (self.rows - 1) * (self.cols - 1)

    def get_legal_actions(self) -> np.ndarray:
        """Returns a boolean mask of length action_space."""
        if self._legal_mask is not None:
            return self._legal_mask
        mask = np.zeros(self.action_space, dtype=bool)
        mask[:self.h_total] = (self.h_edges.flatten() == 0)
        mask[self.h_total:] = (self.v_edges.flatten() == 0)
        self._legal_mask = mask
        return mask

    def _box_complete(self, r: int, c: int) -> bool:
        return (self.h_edges[r, c] == 1 and
                self.h_edges[r + 1, c] == 1 and
                self.v_edges[r, c] == 1 and
                self.v_edges[r, c + 1] == 1)

    def _box_edge_count(self, r: int, c: int) -> int:
        return (self.h_edges[r, c] + self.h_edges[r + 1, c] +
                self.v_edges[r, c] + self.v_edges[r, c + 1])

    def _adjacent_boxes(self, last_typ: str, r: int, c: int) -> List[Tuple[int, int]]:
        boxes = []
        if last_typ == 'h':
            if r > 0:
                boxes.append((r - 1, c))
            if r < self.rows - 1:
                boxes.append((r, c))
        else:
            if c > 0:
                boxes.append((r, c - 1))
            if c < self.cols - 1:
                boxes.append((r, c))
        return boxes

    def _check_new_boxes(self, last_typ: str, r: int, c: int) -> List[Tuple[int, int]]:
        boxes = []
        if last_typ == 'h':
            if r > 0 and self._box_complete(r - 1, c) and self.box_owner[r - 1, c] == -1:
                boxes.append((r - 1, c))
            if r < self.rows - 1 and self._box_complete(r, c) and self.box_owner[r, c] == -1:
                boxes.append((r, c))
        else:  # 'v'
            if c > 0 and self._box_complete(r, c - 1) and self.box_owner[r, c - 1] == -1:
                boxes.append((r, c - 1))
            if c < self.cols - 1 and self._box_complete(r, c) and self.box_owner[r, c] == -1:
                boxes.append((r, c))
        return boxes

    def _count_open3_total(self) -> int:
        """Count all boxes with exactly 3 sides filled and no owner."""
        open3 = 0
        for br in range(self.rows - 1):
            for bc in range(self.cols - 1):
                if self.box_owner[br, bc] == -1 and self._box_edge_count(br, bc) == 3:
                    open3 += 1
        return int(open3)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Place an edge for current_player."""
        typ, r, c = self.action_to_edge(action)
        if typ == 'h':
            if self.h_edges[r, c] == 1:
                return self.get_state_vector(), -1.0, self.is_done(), {"illegal": True}
            self.h_edges[r, c] = 1
            self.h_owner[r, c] = self.current_player
        else:
            if self.v_edges[r, c] == 1:
                return self.get_state_vector(), -1.0, self.is_done(), {"illegal": True}
            self.v_edges[r, c] = 1
            self.v_owner[r, c] = self.current_player

        self._legal_mask = None

        new_boxes = self._check_new_boxes(typ, r, c)

        open3_adjacent = 0
        for br, bc in self._adjacent_boxes(typ, r, c):
            if self.box_owner[br, bc] == -1 and self._box_edge_count(br, bc) == 3:
                open3_adjacent += 1

        reward = 0.0
        turn_passed = False
        if new_boxes:
            for (br, bc) in new_boxes:
                self.box_owner[br, bc] = self.current_player
            self.scores[self.current_player] += len(new_boxes)

            for _ in range(len(new_boxes)):
                idx = self._turn_combo_count 
                reward += self.base_box_reward * (self.combo_multiplier ** idx)
                self._turn_combo_count += 1
        else:
            self._turn_combo_count = 0
            self.current_player = (self.current_player + 1) % self.n_players
            turn_passed = True

        done = self.is_done()

        # Compute total open3 AFTER applying new boxes/owners
        open3_total = self._count_open3_total()

        info = {
            "new_boxes": len(new_boxes),
            "open3_adjacent": int(open3_adjacent),
            "open3_total": int(open3_total),
            "last_edge": (typ, r, c),
            "turn_passed": turn_passed,
        }
        return self.get_state_vector(), reward, done, info

    # State vector for DQN
    def get_state_vector(self) -> np.ndarray:
        h_flat = self.h_edges.flatten()
        v_flat = self.v_edges.flatten()
        edges = np.concatenate([h_flat, v_flat]).astype(np.float32)  # 60

        cp = np.zeros(4, dtype=np.float32)
        cp[self.current_player] = 1.0

        npoh = np.zeros(3, dtype=np.float32)
        npoh[self.n_players - 2] = 1.0

        return np.concatenate([edges, cp, npoh], dtype=np.float32)

    # Utility for GUI picking from mouse
    def nearest_edge_from_point(self, x: float, y: float, w: int, h: int, margin: int) -> Optional[int]:
        gx = (w - 2 * margin) / (self.cols - 1)
        gy = (h - 2 * margin) / (self.rows - 1)

        r_f = int(np.clip(round((y - margin) / gy), 0, self.rows - 1))
        c_f = int(np.clip(round((x - margin) / gx), 0, self.cols - 1))
        candidates = []
        if 0 <= c_f - 1 < self.cols - 1:
            candidates.append(('h', r_f, c_f - 1))
        if 0 <= c_f < self.cols - 1:
            candidates.append(('h', r_f, c_f))
        if 0 <= r_f - 1 < self.rows - 1:
            candidates.append(('v', r_f - 1, c_f))
        if 0 <= r_f < self.rows - 1:
            candidates.append(('v', r_f, c_f))

        def seg_mid(typ, r, c):
            if typ == 'h':
                p1 = (margin + c * gx, margin + r * gy)
                p2 = (margin + (c + 1) * gx, margin + r * gy)
            else:
                p1 = (margin + c * gx, margin + r * gy)
                p2 = (margin + c * gx, margin + (r + 1) * gy)
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        nearest = None
        best_d2 = 1e18
        for (typ, r, c) in candidates:
            if typ == 'h' and self.h_edges[r, c] == 1:
                continue
            if typ == 'v' and self.v_edges[r, c] == 1:
                continue
            mx, my = seg_mid(typ, r, c)
            d2 = (mx - x) ** 2 + (my - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                nearest = (typ, r, c)

        if nearest is None:
            return None
        return self.edge_to_action(*nearest)

    def scores_rank(self) -> List[int]:
        return list(np.argsort(-self.scores))
