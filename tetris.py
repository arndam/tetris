import pygame
import random
from colors import Colors
from block import Block
from tetris_ai import TetrisAI
from config import Config
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

class TetrisRenderer:
    def __init__(self, screen, game):
        self.screen = screen
        self.game = game
        self.font = pygame.font.Font(None, 36)

    def draw_frame(self):
        self.screen.fill(Colors.BLACK)
        
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

    def _get_info_items(self):
        items = [
            "Controls:",
            "←→     Move Block",
            "↑       Rotate Block",
            "↓       Speed Up",
            "SPACE   Hard Drop",
            "P       Pause Game",
            "ESC     Quit Game",
            "A       AI Mode"
        ]
        
        items.extend([
            "",
            "Game Info:",
            f"Level: {self.game.level}",
            f"Score: {self.game.score}",
            f"Lines: {self.game.lines_cleared}",
            f"High Score: {self.game.high_score}"
        ])
        
        if self.game.ai_enabled:
            items.append("AI Mode: ON")
        else:
            items.append("AI Mode: OFF")
        
        return items

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

class TetrisGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris with AI")
        self.clock = pygame.time.Clock()
        self.renderer = TetrisRenderer(self.screen, self)
        self.reset_game()
        
    def reset_game(self):
        # Initialize game area rectangle
        self.game_area = pygame.Rect(
            Config.GAME_AREA_LEFT, 
            Config.GAME_AREA_TOP, 
            Config.GRID_WIDTH * Config.GRID_SIZE, 
            Config.GRID_HEIGHT * Config.GRID_SIZE
        )
        
        # Initialize game state
        self.grid = [[0 for _ in range(Config.GRID_WIDTH)] for _ in range(Config.GRID_HEIGHT)]
        self.current_block = Block()
        self.next_blocks = [Block() for _ in range(Config.PREVIEW_BLOCKS)]
        self.game_over = False
        self.paused = False
        self.ai = TetrisAI()
        self.ai_enabled = False
        
        # Game metrics
        self.score = 0
        self.level = 1
        self.lines_cleared = 0
        self.high_score = self.load_high_score()
        
        # Game timing
        self.drop_speed = 500  # ms
        self.gravity_timer = 0
        self.move_delay = 100  # ms
        self.move_timer = 0
        
    def handle_input(self, event):
        if event.type == pygame.KEYDOWN:
            # Game control keys
            if event.key == pygame.K_ESCAPE:
                self.save_high_score()
                return True  # Signal to exit the game
                
            if event.key == pygame.K_p and not self.game_over:
                self.paused = not self.paused
                
            if event.key == pygame.K_a:
                # Toggle AI
                self.ai_enabled = not self.ai_enabled
                
            if event.key == pygame.K_r and self.game_over:
                self.save_high_score()
                self.reset_game()
                
            # Block movement and rotation (only if game is active)
            if not self.paused and not self.game_over and not self.ai_enabled:
                if event.key == pygame.K_LEFT:
                    self.move_block(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    self.move_block(1, 0)
                elif event.key == pygame.K_DOWN:
                    self.move_block(0, 1)
                elif event.key == pygame.K_UP:
                    self.rotate_block()
                elif event.key == pygame.K_SPACE:
                    # Hard drop
                    while self.is_valid_position():
                        self.current_block.y += 1
                    self.current_block.y -= 1
                    self.move_block(0, 1)
                    
        return False

    def game_loop(self):
        running = True
        last_time = pygame.time.get_ticks()
        
        while running:
            current_time = pygame.time.get_ticks()
            dt = current_time - last_time
            last_time = current_time
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if self.handle_input(event):
                    running = False
                    break
            
            if not running:
                break
                
            if self.game_over or self.paused:
                self.renderer.draw_frame()
                continue
            
            # AI move if enabled
            if self.ai_enabled and not self.game_over:
                self.apply_ai_move()
            
            # Update game state
            self.update(dt)
            
            # Draw the frame
            self.renderer.draw_frame()
            
            # Cap frame rate
            self.clock.tick(Config.FPS)
        
        # Save high score before quitting
        self.save_high_score()
        pygame.quit()

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

    def update(self, dt):
        if self.paused or self.game_over:
            return

        self.gravity_timer += dt
        if self.gravity_timer >= self.drop_speed:
            self.move_block(0, 1)
            self.gravity_timer = 0

        self.high_score = max(self.score, self.high_score)

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
                    self.game_over = True
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
            self.lines_cleared += lines_cleared
            self.level = (self.lines_cleared // 10) + 1
            
            points = {1: 40, 2: 100, 3: 300, 4: 1200}
            self.score += points.get(lines_cleared, 0) * self.level

    def new_block(self):
        if not self.next_blocks:
            # Initialize with the configured number of preview blocks if empty
            self.next_blocks = [Block() for _ in range(Config.PREVIEW_BLOCKS)]
        
        self.current_block = self.next_blocks.pop(0)
        self.next_blocks.append(Block())  # Add new block to end

    def apply_ai_move(self):
        """Apply the best move as determined by the AI"""
        try:
            # Get best move from AI
            best_move = self.ai.get_best_move(self, self.current_block, self.next_blocks)
            
            if best_move is None:
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
                
        except Exception as e:
            print(f"Error in AI move: {e}")
            self.ai_enabled = False

if __name__ == "__main__":
    game = TetrisGame()
    game.game_loop() 