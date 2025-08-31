# config.py
from dataclasses import dataclass

# Board / Game
BOARD_ROWS = 6  # dots in Y
BOARD_COLS = 6  # dots in X
EDGE_COUNT = BOARD_ROWS * (BOARD_COLS - 1) + (BOARD_ROWS - 1) * BOARD_COLS  # 60 for 6x6

# Reward shaping
BASE_BOX_REWARD = 1.0
COMBO_MULTIPLIER = 1.6

# UI / Colors
PLAYER_COLORS = [
    (239, 68, 68),    # red
    (59, 130, 246),   # blue
    (16, 185, 129),   # green
    (234, 179, 8),    # yellow
]

WINDOW_W, WINDOW_H = 1400, 960
CANVAS_MARGIN = 120
DOT_RADIUS = 7

BG_COLOR = (255, 255, 255)
PANEL_COLOR = (242, 242, 247)
TEXT_COLOR = (20, 20, 23)
ACCENT = (37, 99, 235)

EDGE_IDLE = (190, 190, 195)
EDGE_HOVER = (50, 50, 55)
BOX_FILL_ALPHA = 70

# Fonts
FONT_NAME = None
TITLE_SIZE = 56
UI_SIZE = 26
SMALL_SIZE = 20

# Training defaults 
@dataclass
class TrainConfig:
    episodes: int = 3000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 1e-3
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 2000
    target_update: int = 1000
    buffer_size: int = 100_000
    seed: int = 42
    players: int = 2
    # Opponent schedule: "random", "greedy", "alternate" (even/odd episodes), "curriculum" (random -> greedy)
    opponent: str = "random"
    curriculum_switch_ep: int = 1000  # when opponent == "curriculum", switch to greedy at this episode
    combo_multiplier: float = COMBO_MULTIPLIER
    base_box_reward: float = BASE_BOX_REWARD
    open_box_penalty: float = 1.0
    save_path: str = "models/dqn_dots_and_boxes.pth"
    start_player_random: bool = True
