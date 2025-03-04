import pygame
import random
from colors import Colors
from block import Block
from tetris_ai import TetrisAI
from tetris_dqn import TetrisTrainer
from config import Config
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import torch

# CUDA Diagnostics
print("CUDA Diagnostics:")
print(f"CUDA is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name()}")
else:
    print("Possible reasons CUDA is not available:")
    print("1. No NVIDIA GPU detected")
    print("2. CUDA toolkit not installed")
    print("3. Wrong PyTorch version (CPU-only version installed)")

class TetrisRenderer:
    def __init__(self, screen, game):
        self.screen = screen
        self.game = game
        self.font = pygame.font.Font(None, 36)
        self.training_surface = None

    def draw_frame(self):
        self.screen.fill(Colors.BLACK)
        
        if not self.game.training or self.game.trainer.episodes_completed % 100 == 0:
            self.draw_grid()
            self.draw_current_block()
            self.draw_next_block()
        
        self.draw_ui()
        
        if self.game.game_over:
            self.draw_game_over()
        elif self.game.paused:
            self.draw_pause_screen()
            
        pygame.display.flip()

    def draw_grid(self):
        for y in range(Config.GRID_HEIGHT):
            for x in range(Config.GRID_WIDTH):
                cell_rect = pygame.Rect(
                    self.game.game_area.left + x * Config.GRID_SIZE,
                    self.game.game_area.top + y * Config.GRID_SIZE,
                    Config.GRID_SIZE - 1,
                    Config.GRID_SIZE - 1
                )
                if self.game.grid[y][x]:
                    pygame.draw.rect(self.screen, Colors.CYAN, cell_rect)
                else:
                    pygame.draw.rect(self.screen, Colors.GRAY, cell_rect, 1)

    def draw_current_block(self):
        if self.game.current_block:
            for y, row in enumerate(self.game.current_block.shape):
                for x, cell in enumerate(row):
                    if cell:
                        cell_rect = pygame.Rect(
                            self.game.game_area.left + (self.game.current_block.x + x) * Config.GRID_SIZE,
                            self.game.game_area.top + (self.game.current_block.y + y) * Config.GRID_SIZE,
                            Config.GRID_SIZE - 1,
                            Config.GRID_SIZE - 1
                        )
                        pygame.draw.rect(self.screen, self.game.current_block.color, cell_rect)

    def draw_next_block(self):
        preview_area = pygame.Rect(
            self.game.game_area.right + 50,
            self.game.game_area.top,
            150,
            150 * Config.PREVIEW_BLOCKS  # Height scales with number of preview blocks
        )
        pygame.draw.rect(self.screen, Colors.GRAY, preview_area, 2)
        
        next_text = self.font.render("Next:", True, Colors.WHITE)
        self.screen.blit(next_text, (preview_area.left, preview_area.top - 40))
        
        if self.game.next_blocks:
            block_preview_size = Config.GRID_SIZE * 0.8
            
            # Draw each preview block
            for i, next_block in enumerate(self.game.next_blocks):
                # Calculate vertical position for each block
                center_offset_x = (150 - len(next_block.shape[0]) * block_preview_size) / 2
                center_offset_y = ((i * 150) + 75 - len(next_block.shape) * block_preview_size) / 2
                
                for y, row in enumerate(next_block.shape):
                    for x, cell in enumerate(row):
                        if cell:
                            cell_rect = pygame.Rect(
                                preview_area.left + center_offset_x + x * block_preview_size,
                                preview_area.top + center_offset_y + y * block_preview_size,
                                block_preview_size - 1,
                                block_preview_size - 1
                            )
                            pygame.draw.rect(self.screen, next_block.color, cell_rect)

    def draw_ui(self):
        left_margin = 50
        top_margin = 50
        line_spacing = 40
        
        info_items = self._get_info_items()
        total_height = len(info_items) * line_spacing
        padding = 20
        
        controls_bg = pygame.Rect(
            left_margin - 10,
            top_margin - 10,
            300,
            total_height + padding
        )
        pygame.draw.rect(self.screen, Colors.GRAY, controls_bg, 2)
        
        for i, text in enumerate(info_items):
            y_pos = top_margin + (i * line_spacing)
            text_surface = self.font.render(text, True, Colors.WHITE)
            self.screen.blit(text_surface, (left_margin, y_pos))

        if self.game.training:
            self.draw_training_progress()

    def _get_info_items(self):
        info_items = [
            f"Score: {self.game.score}",
            f"Level: {self.game.level}",
            f"Lines: {self.game.lines_cleared_total}",
            f"High Score: {self.game.high_score}",
            "",
            "Controls:",
            "LEFT   Move Left",
            "RIGHT  Move Right",
            "UP     Rotate",
            "DOWN   Drop faster",
            "SPACE  Drop instantly",
            "P      Pause",
            "ESC    Exit",
            "A      Autoplay",
            "T      Training Mode"
        ]
        
        if self.game.training:
            stats = self.game.trainer.training_stats
            current_episode = self.game.trainer.episodes_completed
            max_score = stats.get('max_score', 0)
            avg_score = stats['avg_scores'][-1] if stats['avg_scores'] else 0
            
            info_items.extend([
                "",
                "Training Status:",
                f"Episode: {current_episode}/{self.game.trainer.training_episodes}",
                f"Progress: {(current_episode/self.game.trainer.training_episodes)*100:.1f}%",
                f"Current Score: {self.game.score}",
                f"Max Score: {max_score}",
                f"Avg Score: {avg_score:.1f}",
                "",
                "Press T to Stop Training"
            ])
        
        return info_items

    def draw_game_over(self):
        overlay = pygame.Surface((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
        overlay.fill(Colors.BLACK)
        overlay.set_alpha(160)
        self.screen.blit(overlay, (0, 0))
        
        texts = [
            self.font.render("GAME OVER!", True, Colors.WHITE),
            self.font.render(f"Final Score: {self.game.score}", True, Colors.WHITE),
            self.font.render(f"High Score: {self.game.high_score}", True, Colors.WHITE),
            self.font.render("Press R to Restart", True, Colors.WHITE),
            self.font.render("Press ESC to Exit", True, Colors.WHITE)
        ]
        
        max_width = max(text.get_width() for text in texts)
        padding = 40
        line_spacing = 50
        text_height = line_spacing * len(texts)
        
        bg_rect = pygame.Rect(
            Config.WINDOW_WIDTH // 2 - (max_width + padding) // 2,
            Config.WINDOW_HEIGHT // 2 - (text_height + padding) // 2,
            max_width + padding,
            text_height + padding
        )
        
        pygame.draw.rect(self.screen, Colors.BLACK, bg_rect)
        pygame.draw.rect(self.screen, Colors.WHITE, bg_rect, 2)
        
        for i, text in enumerate(texts):
            text_rect = text.get_rect(
                center=(
                    Config.WINDOW_WIDTH // 2,
                    bg_rect.top + padding // 2 + i * line_spacing
                )
            )
            self.screen.blit(text, text_rect)

    def draw_pause_screen(self):
        overlay = pygame.Surface((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
        overlay.fill(Colors.BLACK)
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        pause_text = self.font.render("PAUSED", True, Colors.WHITE)
        pause_rect = pause_text.get_rect(center=(Config.WINDOW_WIDTH // 2, Config.WINDOW_HEIGHT // 2))
        
        bg_rect = pause_rect.inflate(40, 20)
        pygame.draw.rect(self.screen, Colors.BLACK, bg_rect)
        pygame.draw.rect(self.screen, Colors.WHITE, bg_rect, 2)
        
        self.screen.blit(pause_text, pause_rect)
        
        resume_text = self.font.render("Press P to Resume", True, Colors.WHITE)
        resume_rect = resume_text.get_rect(center=(Config.WINDOW_WIDTH // 2, Config.WINDOW_HEIGHT // 2 + 50))
        self.screen.blit(resume_text, resume_rect)

    def draw_training_progress(self):
        if not self.training_surface:
            self.training_surface = pygame.Surface((400, 300))
        
        fig = Figure(figsize=(4, 3), dpi=100)
        canvas = FigureCanvasAgg(fig)
        
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        episodes = self.game.trainer.training_stats['episodes']
        scores = self.game.trainer.training_stats['scores']
        avg_scores = self.game.trainer.training_stats['avg_scores']
        
        if episodes:
            ax1.plot(episodes, scores, 'b-', alpha=0.3, label='Score')
            ax1.plot(episodes, avg_scores, 'r-', label='Avg Score')
            ax1.set_title('Training Scores')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            progress = (self.game.trainer.episodes_completed / self.game.trainer.training_episodes) * 100
            ax2.barh(['Progress'], [progress], color='green')
            ax2.barh(['Progress'], [100], color='gray', alpha=0.3)
            ax2.set_xlim(0, 100)
            ax2.set_title(f'Training Progress: {progress:.1f}%')
        
        fig.tight_layout()
        canvas.draw()
        
        buf = np.asarray(canvas.buffer_rgba())
        buf = buf[:, :, :3]
        surf = pygame.surfarray.make_surface(buf.swapaxes(0, 1))
        
        if surf.get_size() != (400, 300):
            surf = pygame.transform.scale(surf, (400, 300))
        
        self.screen.blit(surf, (700, 50))

class TetrisGame:
    def __init__(self):
        self.game_area = pygame.Rect(
            Config.GAME_AREA_LEFT,
            Config.GAME_AREA_TOP,
            Config.GRID_WIDTH * Config.GRID_SIZE,
            Config.GRID_HEIGHT * Config.GRID_SIZE
        )
        
        self.grid = [[0 for _ in range(Config.GRID_WIDTH)] for _ in range(Config.GRID_HEIGHT)]
        self.current_block = None
        self.next_blocks = []  # List to hold next blocks
        self.game_over = False
        self.paused = False
        self.score = 0
        self.high_score = self.load_high_score()
        self.level = 1
        self.lines_cleared_total = 0
        self.fall_speed = 1000
        self.fall_time = 0
        self.last_fall_time = pygame.time.get_ticks()
        self.autoplay = False
        self.ai = TetrisAI()
        self.move_delay = 0
        self.last_move_time = 0
        self.trainer = TetrisTrainer(device=Config.DEVICE)
        self.training = False
        self.training_episodes = Config.TRAINING_EPISODES
        self.training_progress = 0
        
        self.reset()

    def reset(self):
        self.grid = [[0 for _ in range(Config.GRID_WIDTH)] for _ in range(Config.GRID_HEIGHT)]
        self.current_block = None
        self.next_blocks = []
        self.game_over = False
        self.paused = False
        self.score = 0
        self.level = 1
        self.lines_cleared_total = 0
        self.fall_speed = 1000
        self.fall_time = 0
        self.last_fall_time = pygame.time.get_ticks()
        self.autoplay = False
        self.move_delay = 0
        self.last_move_time = 0

    def new_block(self):
        if not self.next_blocks:
            # Initialize with the configured number of preview blocks if empty
            self.next_blocks = [Block() for _ in range(Config.PREVIEW_BLOCKS)]
        
        self.current_block = self.next_blocks.pop(0)
        self.next_blocks.append(Block())  # Add new block to end

    def move_block(self, dx, dy):
        if not self.current_block:
            return False
        
        self.current_block.x += dx
        self.current_block.y += dy
        
        if not self.is_valid_position():
            self.current_block.x -= dx
            self.current_block.y -= dy
            if dy > 0:  # Block has landed
                self.merge_block()
                self.clear_lines()
                self.new_block()
                if not self.is_valid_position():
                    # Only set game_over if not in training mode
                    # In training mode, we want to continue and learn from mistakes
                    if not self.training:
                        self.game_over = True
                    else:
                        # In training mode, just clear the top rows to make space
                        # This allows the agent to continue learning
                        self.clear_top_rows(4)  # Clear top 4 rows to make space
                        # Move the current block up to ensure it's in a valid position
                        while not self.is_valid_position() and self.current_block.y > 0:
                            self.current_block.y -= 1
            return False
        return True

    def rotate_block(self):
        if not self.current_block:
            return
        self.current_block.rotate()
        if not self.is_valid_position():
            self.current_block.rotate_back()

    def is_valid_position(self):
        for y, row in enumerate(self.current_block.shape):
            for x, cell in enumerate(row):
                if cell:
                    pos_x = self.current_block.x + x
                    pos_y = self.current_block.y + y
                    if (pos_x < 0 or pos_x >= Config.GRID_WIDTH or
                        pos_y >= Config.GRID_HEIGHT or pos_y < 0 or
                        (pos_y >= 0 and self.grid[pos_y][pos_x])):
                        return False
        return True

    def merge_block(self):
        if not self.current_block:
            return
        
        for y, row in enumerate(self.current_block.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = self.current_block.y + y
                    grid_x = self.current_block.x + x
                    if 0 <= grid_y < Config.GRID_HEIGHT and 0 <= grid_x < Config.GRID_WIDTH:
                        self.grid[grid_y][grid_x] = 1

    def clear_lines(self):
        lines_cleared = 0
        y = Config.GRID_HEIGHT - 1
        while y >= 0:
            if all(self.grid[y]):
                del self.grid[y]
                self.grid.insert(0, [0 for _ in range(Config.GRID_WIDTH)])
                lines_cleared += 1
            else:
                y -= 1
                
        if lines_cleared > 0:
            self.lines_cleared_total += lines_cleared
            self.level = (self.lines_cleared_total // 10) + 1
            
            points = {1: 40, 2: 100, 3: 300, 4: 1200}
            self.score += points.get(lines_cleared, 0) * self.level

    def handle_autoplay(self, current_time):
        if not self.autoplay or self.paused or self.game_over:
            return

        if current_time - self.last_move_time < self.move_delay:
            return
        
        try:
            # Get best move from AI
            best_move = self.ai.get_best_move(self, self.current_block, self.next_blocks)
            
            if best_move is None:
                print("AI couldn't find a valid move")
                return
                
            rotation, dx = best_move
            
            # Save original position
            original_x = self.current_block.x
            original_y = self.current_block.y
            original_shape = [row[:] for row in self.current_block.shape]
            
            try:
                # Execute rotations
                for _ in range(rotation):
                    self.rotate_block()
                
                # Execute horizontal movement
                if dx < 0:
                    for _ in range(abs(dx)):
                        self.move_block(-1, 0)
                else:
                    for _ in range(dx):
                        self.move_block(1, 0)
                
                # Drop the piece
                while self.is_valid_position():
                    self.current_block.y += 1
                self.current_block.y -= 1
                self.move_block(0, 1)
            except Exception as e:
                # If any error occurs during move execution, restore original position
                print(f"Error executing move: {e}")
                self.current_block.x = original_x
                self.current_block.y = original_y
                self.current_block.shape = original_shape
            
            self.last_move_time = current_time
            self.move_delay = 50  # Faster autoplay speed
            
        except Exception as e:
            print(f"Error in autoplay: {e}")
            self.autoplay = False

    def load_high_score(self):
        try:
            with open('highscore.txt', 'r') as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            with open('highscore.txt', 'w') as f:
                f.write('0')
            return 0
            
    def save_high_score(self):
        try:
            # Update high score if current score is higher
            if self.score > self.high_score:
                self.high_score = self.score
            
            # Always write the current high score to file
            with open('highscore.txt', 'w') as f:
                f.write(str(self.high_score))
        except Exception as e:
            print(f"Error saving high score: {e}")

    def update(self, current_time):
        if self.paused or self.game_over:
            return

        delta_time = current_time - self.last_fall_time
        self.fall_speed = max(50, 1000 - (self.level * 50))

        if delta_time >= self.fall_speed:
            self.move_block(0, 1)
            self.last_fall_time = current_time

        self.high_score = max(self.score, self.high_score)

    def clear_top_rows(self, num_rows):
        """Clear the top rows to make space during training"""
        for i in range(min(num_rows, Config.GRID_HEIGHT)):
            self.grid[i] = [0 for _ in range(Config.GRID_WIDTH)]

class Tetris:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        
        self.game = TetrisGame()
        self.renderer = TetrisRenderer(self.screen, self.game)

    def _handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            self.game.save_high_score()
            if self.game.training:
                self.game.trainer.save_model()
            return True
        if key == pygame.K_r and self.game.game_over:
            self.game.save_high_score()
            self.game.reset()
            self.game.new_block()
        if key == pygame.K_p and not self.game.game_over:
            self.game.paused = not self.game.paused
        if key == pygame.K_a and not self.game.game_over:
            self.game.autoplay = not self.game.autoplay
        if not self.game.paused and not self.game.game_over:
            if key == pygame.K_LEFT:
                self.game.move_block(-1, 0)
            elif key == pygame.K_RIGHT:
                self.game.move_block(1, 0)
            elif key == pygame.K_DOWN:
                self.game.move_block(0, 1)
            elif key == pygame.K_UP:
                self.game.rotate_block()
            elif key == pygame.K_SPACE:
                while self.game.is_valid_position():
                    self.game.current_block.y += 1
                self.game.current_block.y -= 1
                self.game.move_block(0, 1)
        if key == pygame.K_t and not self.game.game_over:
            self.game.training = not self.game.training
            if self.game.training:
                self.game.trainer.set_training_episodes(self.game.training_episodes)
        return False

    def run(self):
        self.game.new_block()
        
        try:
            while True:
                current_time = pygame.time.get_ticks()
                self.clock.tick(Config.FPS)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.game.save_high_score()
                        if self.game.training:
                            self.game.trainer.save_model()
                        return
                    if event.type == pygame.KEYDOWN:
                        if self._handle_keydown(event.key):
                            return

                # Handle training mode separately from normal gameplay
                if self.game.training and not self.game.paused:
                    # Let the trainer handle everything during training
                    self.game.trainer.train_step(self.game)
                    
                    # Only render occasionally to speed up training
                    if self.game.trainer.episodes_completed % 100 == 0 or self.game.trainer.current_step % 1000 == 0:
                        self.renderer.draw_frame()
                    
                    # Skip the rest of the game loop during training
                    continue

                # Normal gameplay (non-training)
                self.game.handle_autoplay(current_time)
                self.game.update(current_time)
                self.renderer.draw_frame()

        finally:
            if self.game.training:
                self.game.trainer.save_model()

if __name__ == "__main__":
    game = Tetris()
    game.run() 