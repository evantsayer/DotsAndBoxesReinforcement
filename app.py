# app.py
import os
import sys
import threading
import pygame
import numpy as np

from config import *
from dots_and_boxes import DotsAndBoxesEnv
from agents import RandomAgent, GreedyAgent, DQNAgent, list_model_files
from train import TrainConfig, train_dqn

try:
    import analyze
except Exception:
    analyze = None

pygame.init()
pygame.display.set_caption("Dots & Boxes — RL Showcase")

# UI Helpers
def load_font(size):
    return pygame.font.Font(FONT_NAME, size)

FONT_TITLE = load_font(TITLE_SIZE)
FONT_UI = load_font(UI_SIZE)
FONT_SMALL = load_font(SMALL_SIZE)

def draw_text(surface, text, x, y, font, color=TEXT_COLOR, center=False):
    s = font.render(text, True, color)
    r = s.get_rect()
    if center:
        r.center = (x, y)
    else:
        r.topleft = (x, y)
    surface.blit(s, r)

class Button:
    def __init__(self, rect, text, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.hover = False

    def handle(self, event, block=False):
        if block:
            return
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.callback()

    def draw(self, surf):
        color = (220, 222, 228) if not self.hover else (204, 206, 212)
        pygame.draw.rect(surf, color, self.rect, border_radius=12)
        pygame.draw.rect(surf, (190, 192, 196), self.rect, width=2, border_radius=12)
        draw_text(surf, self.text, self.rect.centerx, self.rect.centery, FONT_UI, color=(20,20,23), center=True)

class Toggle:
    def __init__(self, rect, label, initial=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = initial

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.value = not self.value

    def draw(self, surf):
        pygame.draw.rect(surf, (220, 222, 228), self.rect, border_radius=10)
        pygame.draw.rect(surf, (190, 192, 196), self.rect, width=2, border_radius=10)
        knob_r = self.rect.height - 8
        x = self.rect.x + 4 + (self.rect.width - knob_r - 8 if self.value else 0)
        pygame.draw.rect(surf, (100,100,110) if not self.value else ACCENT,
                         (x, self.rect.y+4, knob_r, knob_r), border_radius=8)
        draw_text(surf, f"{self.label}: {'ON' if self.value else 'OFF'}",
                  self.rect.right + 16, self.rect.y + self.rect.height//2 - 12, FONT_UI)

class TextInput:
    def __init__(self, rect, text=""):
        self.rect = pygame.Rect(rect)
        self.text = str(text)
        self.active = False
        self.cursor_visible = True
        self.last_blink = 0
        self.blink_ms = 500

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)
        elif self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.active = False
            else:
                if event.unicode and 32 <= ord(event.unicode) <= 126:
                    self.text += event.unicode

    def draw(self, surf):
        pygame.draw.rect(surf, (255, 255, 255), self.rect, border_radius=8)
        pygame.draw.rect(surf, (190, 192, 196), self.rect, width=2, border_radius=8)
        draw_text(surf, self.text, self.rect.x + 10, self.rect.y + 6, FONT_UI, color=TEXT_COLOR)
        if self.active:
            now = pygame.time.get_ticks()
            if now - self.last_blink > self.blink_ms:
                self.cursor_visible = not self.cursor_visible
                self.last_blink = now
            if self.cursor_visible:
                w, _ = FONT_UI.size(self.text)
                cx = self.rect.x + 10 + w + 2
                cy = self.rect.y + 6
                pygame.draw.rect(surf, (50, 50, 55), (cx, cy, 2, self.rect.height - 12))

class Dropdown:
    def __init__(self, rect, options, initial_index=0):
        self.rect = pygame.Rect(rect)
        self.options = options[:]
        self.index = initial_index
        self.open = False
        self.item_height = self.rect.height

    @property
    def value(self):
        return self.options[self.index] if self.options else ""

    def set_options(self, options, keep_value=True):
        old_val = self.value if keep_value and self.options else None
        self.options = options[:]
        if old_val in self.options:
            self.index = self.options.index(old_val)
        else:
            self.index = 0
        self.open = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
            elif self.open:
                for i, _ in enumerate(self.options):
                    r = pygame.Rect(self.rect.x, self.rect.bottom + i * self.item_height,
                                    self.rect.width, self.item_height)
                    if r.collidepoint(event.pos):
                        self.index = i
                        self.open = False
                        return
                self.open = False

    def draw_base(self, surf):
        pygame.draw.rect(surf, (220, 222, 228), self.rect, border_radius=8)
        pygame.draw.rect(surf, (190, 192, 196), self.rect, width=2, border_radius=8)
        draw_text(surf, str(self.value), self.rect.x + 10, self.rect.y + 6, FONT_UI)
        pygame.draw.polygon(surf, (60,60,65), [
            (self.rect.right-20, self.rect.y+12),
            (self.rect.right-8, self.rect.y+12),
            (self.rect.right-14, self.rect.y+20),
        ])

    def draw_menu(self, surf):
        if not self.open:
            return
        for i, opt in enumerate(self.options):
            r = pygame.Rect(self.rect.x, self.rect.bottom + i * self.item_height,
                            self.rect.width, self.item_height)
            pygame.draw.rect(surf, (255, 255, 255), r)
            pygame.draw.rect(surf, (190, 192, 196), r, width=1)
            draw_text(surf, str(opt), r.x + 10, r.y + 6, FONT_UI)

# Screens
class ScreenBase:
    def handle(self, event): ...
    def update(self, dt): ...
    def draw(self, surf): ...

class MainMenu(ScreenBase):
    def __init__(self, app):
        self.app = app
        self.btn_play = Button((WINDOW_W//2-140, 340, 280, 60), "Play", lambda: app.set_screen(PlaySetup(app)))
        self.btn_train = Button((WINDOW_W//2-140, 420, 280, 60), "Train", lambda: app.set_screen(TrainSetup(app)))
        self.btn_analyze = Button((WINDOW_W//2-140, 500, 280, 60), "Analyze", lambda: app.set_screen(AnalyzeScreen(app)))
        self.btn_quit = Button((WINDOW_W//2-140, 580, 280, 60), "Quit", lambda: sys.exit(0))

    def handle(self, event):
        self.btn_play.handle(event)
        self.btn_train.handle(event)
        self.btn_analyze.handle(event)
        self.btn_quit.handle(event)

    def draw(self, surf):
        surf.fill(BG_COLOR)
        draw_text(surf, "Dots & Boxes", WINDOW_W//2, 160, FONT_TITLE, center=True)
        draw_text(surf, "RL Showcase", WINDOW_W//2, 210, FONT_UI, color=ACCENT, center=True)
        self.btn_play.draw(surf)
        self.btn_train.draw(surf)
        self.btn_analyze.draw(surf)
        self.btn_quit.draw(surf)

class PlaySetup(ScreenBase):
    def __init__(self, app):
        self.app = app
        self.left_x = 80
        self.col2_x = 260
        self.col3_x = 560
        self.row0_y = 160
        self.row_gap = 80

        self.players_dd = Dropdown((self.col2_x, self.row0_y, 160, 36), ["2", "3", "4"], 0)
        self.type_dd = [Dropdown((self.col2_x, self.row0_y + (i+1)*self.row_gap, 220, 36),
                                 ["Human", "AI:Random", "AI:Greedy", "AI:Model"], 0) for i in range(4)]
        model_files = list_model_files()
        self.model_dd = [Dropdown((self.col3_x, self.row0_y + (i+1)*self.row_gap, 420, 36),
                                  model_files if model_files else ["(no .pth models found)"], 0) for i in range(4)]
        self.btn_start = Button((WINDOW_W-360, WINDOW_H-100, 300, 56), "Start Game", self.start_game)
        self.btn_back = Button((60, WINDOW_H-100, 220, 56), "Back", lambda: app.set_screen(MainMenu(app)))

    def any_dropdown_open(self):
        if self.players_dd.open:
            return True
        return any(dd.open for dd in self.type_dd) or any(dd.open for dd in self.model_dd)

    def start_game(self):
        n = int(self.players_dd.value)
        types = [dd.value for dd in self.type_dd[:n]]
        models = [dd.value for dd in self.model_dd[:n]]
        self.app.set_screen(GameScreen(self.app, n, types, models))

    def handle(self, event):
        self.players_dd.handle(event)
        for dd in self.type_dd:
            dd.handle(event)
        for mdd in self.model_dd:
            mdd.handle(event)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_F5:
            files = list_model_files()
            for mdd in self.model_dd:
                mdd.set_options(files if files else ["(no .pth models found)"])

        block = self.any_dropdown_open() and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION)
        self.btn_start.handle(event, block=block)
        self.btn_back.handle(event, block=block)

    def draw(self, surf):
        surf.fill(BG_COLOR)
        draw_text(surf, "Play — Setup", 60, 60, FONT_TITLE)
        draw_text(surf, "Players", self.left_x, self.row0_y + 6, FONT_UI)

        draw_text(surf, "Type", self.col2_x, self.row0_y + self.row_gap - 28, FONT_SMALL)
        draw_text(surf, "Model (AI:Model)", self.col3_x, self.row0_y + self.row_gap - 28, FONT_SMALL)

        self.players_dd.draw_base(surf)
        n = int(self.players_dd.value)
        for i in range(n):
            draw_text(surf, f"P{i+1}", self.left_x, self.row0_y + (i+1)*self.row_gap + 6, FONT_UI,
                      color=PLAYER_COLORS[i])
            self.type_dd[i].draw_base(surf)
            self.model_dd[i].draw_base(surf)

        self.btn_start.draw(surf)
        self.btn_back.draw(surf)

        self.players_dd.draw_menu(surf)
        for i in range(n):
            self.type_dd[i].draw_menu(surf)
            self.model_dd[i].draw_menu(surf)

class TrainSetup(ScreenBase):
    def __init__(self, app):
        self.app = app
        self.left_x = 80
        self.input_x = 280
        self.row_y = 160
        self.gap = 56

        self.episodes = TextInput((self.input_x, self.row_y + 0*self.gap, 200, 38), "3000")
        self.batch = TextInput((self.input_x, self.row_y + 1*self.gap, 200, 38), "256")
        self.lr = TextInput((self.input_x, self.row_y + 2*self.gap, 200, 38), "0.001")
        self.gamma = TextInput((self.input_x, self.row_y + 3*self.gap, 200, 38), "0.99")
        self.eps = TextInput((self.input_x, self.row_y + 4*self.gap, 200, 38), "1.0,0.05,2000")
        self.players = Dropdown((self.input_x, self.row_y + 5*self.gap, 200, 38), ["2","3","4"], 0)
        self.opponent = Dropdown((self.input_x, self.row_y + 6*self.gap, 200, 38),
                                 ["random","greedy","alternate","curriculum"], 0)
        self.curriculum_ep = TextInput((self.input_x, self.row_y + 7*self.gap, 200, 38), "1000")
        self.combo = TextInput((self.input_x, self.row_y + 8*self.gap, 200, 38), f"{COMBO_MULTIPLIER}")
        self.base_reward = TextInput((self.input_x, self.row_y + 9*self.gap, 200, 38), f"{BASE_BOX_REWARD}")
        self.open_penalty = TextInput((self.input_x, self.row_y + 10*self.gap, 200, 38), "1.0")

        self.model_name = TextInput((620, self.row_y + 0*self.gap, 520, 38), "models/dqn_dots_and_boxes.pth")
        self.rand_start = Toggle((620, self.row_y + 1*self.gap, 90, 36), "Randomize start player", initial=True)

        self.btn_start = Button((WINDOW_W-360, WINDOW_H-100, 300, 56), "Start Training", self.start_training)
        self.btn_back = Button((60, WINDOW_H-100, 220, 56), "Back", lambda: app.set_screen(MainMenu(app)))

        self.training = False
        self.train_status = "idle"

    def parse_cfg(self) -> TrainConfig:
        eps_start, eps_end, eps_decay = [float(x.strip()) for x in self.eps.text.split(",")]
        cfg = TrainConfig(
            episodes=int(self.episodes.text),
            batch_size=int(self.batch.text),
            lr=float(self.lr.text),
            gamma=float(self.gamma.text),
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay_episodes=int(eps_decay),
            players=int(self.players.value),
            opponent=str(self.opponent.value),
            curriculum_switch_ep=int(self.curriculum_ep.text),
            combo_multiplier=float(self.combo.text),
            base_box_reward=float(self.base_reward.text),
            open_box_penalty=float(self.open_penalty.text),
            save_path=self.model_name.text.strip(),
            start_player_random=self.rand_start.value,
        )
        return cfg

    def start_training(self):
        if self.training:
            return
        cfg = self.parse_cfg()
        self.training = True
        self.train_status = "starting..."

        def run():
            def cb(info):
                self.train_status = (f"Device {info['device']} — "
                                     f"Opponents {info.get('opponents','?')} — "
                                     f"Episode {info['episode']}/{info['episodes']}  "
                                     f"ep_reward={info['ep_reward']:.2f}  "
                                     f"mean100={info['mean_100']:.2f}  "
                                     f"ε={info['epsilon']:.3f}")
            try:
                path = train_dqn(cfg, progress_cb=cb)
                self.train_status = f"Done. Saved to {path}. CSV in ./csv and plots in ./plots."
            except Exception as e:
                self.train_status = f"Error: {e}"
            finally:
                self.training = False

        threading.Thread(target=run, daemon=True).start()

    def any_dropdown_open(self):
        return self.players.open or self.opponent.open

    def handle(self, event):
        for w in [self.episodes, self.batch, self.lr, self.gamma, self.eps,
                  self.curriculum_ep, self.combo, self.base_reward, self.open_penalty, self.model_name]:
            w.handle(event)
        self.players.handle(event)
        self.opponent.handle(event)
        self.rand_start.handle(event)

        block = self.any_dropdown_open() and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION)
        self.btn_start.handle(event, block=block)
        self.btn_back.handle(event, block=block)

    def draw(self, surf):
        surf.fill(BG_COLOR)
        draw_text(surf, "Train — DQN", 60, 60, FONT_TITLE)

        labels = [
            ("Episodes", self.episodes),
            ("Batch", self.batch),
            ("LR", self.lr),
            ("Gamma", self.gamma),
            ("Epsilon s,e,decay_eps", self.eps),
            ("Players", None),
            ("Opponent schedule", None),
            ("Curriculum switch @ episode", self.curriculum_ep),
            ("Combo multiplier", self.combo),
            ("Base box reward", self.base_reward),
            ("Open-box penalty", self.open_penalty),
        ]

        for i, (txt, _) in enumerate(labels):
            y = self.row_y + i*self.gap + 6
            draw_text(surf, txt, self.left_x, y, FONT_UI)

        for w in [self.episodes, self.batch, self.lr, self.gamma, self.eps,
                  self.curriculum_ep, self.combo, self.base_reward, self.open_penalty, self.model_name]:
            w.draw(surf)
        self.players.draw_base(surf)
        self.opponent.draw_base(surf)
        self.rand_start.draw(surf)

        panel_rect = pygame.Rect(620, self.row_y + 2*self.gap, WINDOW_W - 700, 320)
        pygame.draw.rect(surf, PANEL_COLOR, panel_rect, border_radius=12)
        pygame.draw.rect(surf, (190, 192, 196), panel_rect, width=2, border_radius=12)
        draw_text(surf, "Status", panel_rect.x + 16, panel_rect.y + 12, FONT_UI)
        draw_text(surf, self.train_status, panel_rect.x + 16, panel_rect.y + 56, FONT_SMALL)

        self.btn_start.draw(surf)
        self.btn_back.draw(surf)

        self.players.draw_menu(surf)
        self.opponent.draw_menu(surf)

class AnalyzeScreen(ScreenBase):
    def __init__(self, app):
        self.app = app
        self.btn_run = Button((WINDOW_W//2-180, 420, 360, 60), "Run Aggregate Analysis", self.run_analysis)
        self.btn_back = Button((60, WINDOW_H-100, 220, 56), "Back", lambda: app.set_screen(MainMenu(app)))
        self.status = "Looks for all CSVs in ./csv and writes plots to ./plots/aggregate"
        self.running = False

    def run_analysis(self):
        if self.running:
            return
        if analyze is None:
            self.status = "analyze.py not found. Make sure it's in the project folder."
            return
        self.running = True
        self.status = "Running…"
        def work():
            try:
                out = analyze.run_aggregate()
                if out.get("ok"):
                    self.status = f"Done. Plots in {out.get('out_dir')}  (and csv/aggregate_summary.csv)"
                else:
                    self.status = "No CSVs found. Train a model first."
            except Exception as e:
                self.status = f"Error: {e}"
            finally:
                self.running = False
        threading.Thread(target=work, daemon=True).start()

    def handle(self, event):
        self.btn_run.handle(event)
        self.btn_back.handle(event)

    def draw(self, surf):
        surf.fill(BG_COLOR)
        draw_text(surf, "Aggregate Analysis", WINDOW_W//2, 160, FONT_TITLE, center=True)
        draw_text(surf, "Compare multiple runs by hyperparameters, schedules, and penalties.", WINDOW_W//2, 210, FONT_UI, color=ACCENT, center=True)
        self.btn_run.draw(surf)
        draw_text(surf, self.status, WINDOW_W//2, 520, FONT_SMALL, center=True)
        self.btn_back.draw(surf)

class GameScreen(ScreenBase):
    def __init__(self, app, n_players, type_labels, model_labels):
        self.app = app
        self.bottom_h = 120
        self.env = DotsAndBoxesEnv(n_players=n_players, base_box_reward=BASE_BOX_REWARD, combo_multiplier=COMBO_MULTIPLIER)
        self.types = type_labels
        self.models = model_labels
        self.agents = []
        for i in range(n_players):
            t = self.types[i].lower()
            if t.startswith("human"):
                self.agents.append(None)
            elif "greedy" in t:
                self.agents.append(GreedyAgent())
            elif "model" in t:
                path = self.models[i]
                if os.path.isfile(path):
                    self.agents.append(DQNAgent(path))
                else:
                    self.agents.append(RandomAgent())
            else:
                self.agents.append(RandomAgent())

        self.btn_back = Button((60, WINDOW_H-self.bottom_h+32, 220, 56), "Back", lambda: app.set_screen(MainMenu(app)))
        self.last_ai_move_time = 0
        self.ai_move_delay_ms = 220
        self.game_over = False
        self.hover_action = None

    def handle(self, event):
        self.btn_back.handle(event)
        if self.game_over:
            return
        cp = self.env.current_player
        if self.agents[cp] is None:
            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                self.hover_action = self.env.nearest_edge_from_point(mx, my, WINDOW_W, WINDOW_H-self.bottom_h, CANVAS_MARGIN)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                a = self.env.nearest_edge_from_point(mx, my, WINDOW_W, WINDOW_H-self.bottom_h, CANVAS_MARGIN)
                if a is not None:
                    _, _, done, _ = self.env.step(a)
                    self.game_over = done

    def update(self, dt):
        if self.game_over:
            return
        cp = self.env.current_player
        if self.agents[cp] is not None:
            now = pygame.time.get_ticks()
            if now - self.last_ai_move_time > self.ai_move_delay_ms:
                a = self.agents[cp].select_action(self.env)
                _, _, done, _ = self.env.step(a)
                self.last_ai_move_time = now
                self.game_over = done

    def draw_board(self, surf):
        w, h = WINDOW_W, WINDOW_H - self.bottom_h
        gx = (w - 2 * CANVAS_MARGIN) / (self.env.cols - 1)
        gy = (h - 2 * CANVAS_MARGIN) / (self.env.rows - 1)

        for r in range(self.env.rows - 1):
            for c in range(self.env.cols - 1):
                owner = self.env.box_owner[r, c]
                if owner >= 0:
                    color = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
                    rect = pygame.Rect(CANVAS_MARGIN + c * gx, CANVAS_MARGIN + r * gy, gx, gy)
                    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
                    s.fill((*color, BOX_FILL_ALPHA))
                    surf.blit(s, rect.topleft)

        for r in range(self.env.rows):
            for c in range(self.env.cols - 1):
                x1 = CANVAS_MARGIN + c * gx
                y1 = CANVAS_MARGIN + r * gy
                x2 = CANVAS_MARGIN + (c + 1) * gx
                y2 = y1
                taken = self.env.h_edges[r, c] == 1
                if taken:
                    owner = int(self.env.h_owner[r, c])
                    col = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
                    pygame.draw.line(surf, col, (x1, y1), (x2, y2), 8)
                else:
                    pygame.draw.line(surf, EDGE_IDLE, (x1, y1), (x2, y2), 4)

        for r in range(self.env.rows - 1):
            for c in range(self.env.cols):
                x1 = CANVAS_MARGIN + c * gx
                y1 = CANVAS_MARGIN + r * gy
                x2 = x1
                y2 = CANVAS_MARGIN + (r + 1) * gy
                taken = self.env.v_edges[r, c] == 1
                if taken:
                    owner = int(self.env.v_owner[r, c])
                    col = PLAYER_COLORS[owner % len(PLAYER_COLORS)]
                    pygame.draw.line(surf, col, (x1, y1), (x2, y2), 8)
                else:
                    pygame.draw.line(surf, EDGE_IDLE, (x1, y1), (x2, y2), 4)

        if self.hover_action is not None:
            typ, r, c = self.env.action_to_edge(self.hover_action)
            if typ == 'h' and self.env.h_edges[r, c] == 0:
                x1 = CANVAS_MARGIN + c * gx
                y1 = CANVAS_MARGIN + r * gy
                x2 = CANVAS_MARGIN + (c + 1) * gx
                pygame.draw.line(surf, EDGE_HOVER, (x1, y1), (x2, y1), 6)
            elif typ == 'v' and self.env.v_edges[r, c] == 0:
                x1 = CANVAS_MARGIN + c * gx
                y1 = CANVAS_MARGIN + r * gy
                y2 = CANVAS_MARGIN + (r + 1) * gy
                pygame.draw.line(surf, EDGE_HOVER, (x1, y1), (x1, y2), 6)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                x = CANVAS_MARGIN + c * gx
                y = CANVAS_MARGIN + r * gy
                pygame.draw.circle(surf, (40,40,45), (x, y), DOT_RADIUS)

    def draw(self, surf):
        surf.fill(BG_COLOR)
        self.draw_board(surf)

        pygame.draw.rect(surf, PANEL_COLOR, (0, WINDOW_H - self.bottom_h, WINDOW_W, self.bottom_h))
        pygame.draw.line(surf, (190, 192, 196), (0, WINDOW_H - self.bottom_h), (WINDOW_W, WINDOW_H - self.bottom_h), 2)

        x = 24
        for i, s in enumerate(self.env.scores):
            draw_text(surf, f"P{i+1}: {int(s)}", x, WINDOW_H - self.bottom_h + 22, FONT_UI, color=PLAYER_COLORS[i])
            x += 160
        status = "Game over" if self.game_over else f"Turn: P{self.env.current_player+1}"
        draw_text(surf, status, WINDOW_W - 300, WINDOW_H - self.bottom_h + 22, FONT_UI, color=ACCENT)

        self.btn_back.draw(surf)

class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.active_screen = MainMenu(self)

    def set_screen(self, screen):
        self.active_screen = screen

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                self.active_screen.handle(event)
            if hasattr(self.active_screen, "update"):
                self.active_screen.update(dt)
            self.active_screen.draw(self.screen)
            pygame.display.flip()

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    App().run()
