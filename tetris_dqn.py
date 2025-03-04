import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import json
from datetime import datetime
import math
from config import Config
from block import Block

class TetrisNet(nn.Module):
    def __init__(self, device):
        super(TetrisNet, self).__init__()
        self.device = device
        
        # First, let's identify what shapes we're actually getting
        self.input_height = 20
        self.input_width = 10
        self.extra_features_size = 8  # 7 for piece type + 1 for level
        
        # Convolutional layers for grid processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate conv output size
        self.conv_output_size = 64 * 20 * 10  # channels * height * width
        
        # Linear layers adjusted for actual input size
        self.fc_extra = nn.Linear(self.extra_features_size, 128)  # Use the class variable
        self.fc1 = nn.Linear(self.conv_output_size + 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 40)  # 40 actions: 4 rotations * 10 positions
        
        self.to(device)

    def forward(self, grid, extra_features):
        # Ensure grid has 4 dimensions [batch, channel, height, width]
        if grid.dim() < 4:
            # If we get a 3D tensor [batch, height, width]
            if grid.dim() == 3:
                grid = grid.unsqueeze(1)
            # If we get a 2D tensor [height, width]
            elif grid.dim() == 2:
                grid = grid.unsqueeze(0).unsqueeze(0)
        
        # Get batch size
        batch_size = grid.size(0)
        
        # Process grid
        x = torch.relu(self.conv1(grid))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Ensure extra_features has a batch dimension
        if extra_features.dim() == 1:
            extra_features = extra_features.unsqueeze(0)
        
        # Process extra features (which has 8 features)
        extra = torch.relu(self.fc_extra(extra_features))
        
        # Ensure extra has the same batch size
        if extra.size(0) != batch_size:
            extra = extra.expand(batch_size, -1)
        
        # Combine
        x = torch.cat((x, extra), dim=1)
        
        # Final layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

    def _debug_shapes(self, grid, extra_features):
        """Helper method to debug tensor shapes"""
        print(f"Grid shape: {grid.shape}")
        print(f"Extra features shape: {extra_features.shape}")
        print(f"Grid device: {grid.device}")
        print(f"Extra features device: {extra_features.device}")

class ReplayBuffer:
    def __init__(self, capacity=10000, device=None):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
    def push(self, state, action, reward, next_state, done):
        # Ensure everything is on the correct device
        grid_state, extra_features = state
        next_grid_state, next_extra_features = next_state
        
        state = (
            grid_state.to(self.device),
            extra_features.to(self.device)
        )
        action = action.to(self.device)
        next_state = (
            next_grid_state.to(self.device),
            next_extra_features.to(self.device)
        )
        
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TetrisTrainer:
    def __init__(self, save_dir='models', device=None):
        self.device = device if device is not None else torch.device("cpu")
        print(f"Training on device: {self.device}")
        
        # Initialize networks with device
        self.policy_net = TetrisNet(self.device)
        self.target_net = TetrisNet(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0005)  # Slightly lower learning rate
        self.memory = ReplayBuffer(capacity=20000, device=self.device)  # Larger replay buffer
        
        self.batch_size = 128
        self.gamma = 0.95  # Slightly lower discount factor to focus more on immediate rewards
        self.eps_start = 1.0  # Start with 100% exploration
        self.eps_end = 0.05  # End with 5% exploration (increased from 1%)
        self.eps_decay = 10000  # Slower decay for better exploration
        self.target_update = 10  # Less frequent target network updates
        self.steps_done = 0
        self.training_episodes = 2000
        self.current_episode = 0
        self.current_step = 0
        self.episodes_completed = 0
        
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.training_stats = {
            'episodes': [],
            'scores': [],
            'avg_scores': [],
            'max_score': 0,
            'training_time': 0
        }
        
        self.load_model()

    def get_state(self, game):
        try:
            # Ensure the grid is valid
            if not game.grid or len(game.grid) != Config.GRID_HEIGHT or len(game.grid[0]) != Config.GRID_WIDTH:
                print("Invalid grid state, reinitializing")
                game.grid = [[0 for _ in range(Config.GRID_WIDTH)] for _ in range(Config.GRID_HEIGHT)]
            
            # Convert game state to tensor and move to device
            grid_state = torch.FloatTensor(game.grid).to(self.device)
            
            # One-hot encode next piece (use the first piece from next_blocks)
            next_piece = np.zeros(7)
            if game.next_blocks and len(game.next_blocks) > 0:
                if hasattr(game.next_blocks[0], 'shape_index'):
                    next_piece[game.next_blocks[0].shape_index] = 1
                else:
                    print("Next block missing shape_index, using default")
            
            extra_features = np.concatenate([
                next_piece,
                [game.level]
            ])
            
            extra_features = torch.FloatTensor(extra_features).to(self.device)
            return grid_state, extra_features
        except Exception as e:
            print(f"Error in get_state: {e}")
            import traceback
            traceback.print_exc()
            # Return a default state if there's an error
            default_grid = torch.zeros((Config.GRID_HEIGHT, Config.GRID_WIDTH)).to(self.device)
            default_extra = torch.zeros(8).to(self.device)  # 7 for piece type + 1 for level
            return default_grid, default_extra

    def select_action(self, state, training=False):
        grid_state, extra_features = state
        
        if training:
            # Calculate exploration probability
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            
            # Print exploration rate occasionally
            if self.current_step % 1000 == 0:
                print(f"Current exploration rate: {eps_threshold:.4f}")
            
            # Exploration: random action
            if random.random() < eps_threshold:
                # More intelligent exploration - bias toward center positions
                rotation = random.randrange(4)
                
                # Prefer positions in the middle of the board
                position_weights = [0.5, 0.7, 0.9, 1.0, 1.0, 1.0, 0.9, 0.7, 0.5, 0.3]
                position_weights = position_weights[:Config.GRID_WIDTH]
                position = random.choices(
                    range(Config.GRID_WIDTH), 
                    weights=position_weights, 
                    k=1
                )[0]
                
                action = rotation * Config.GRID_WIDTH + position
                return torch.tensor([[action]], device=self.device)
            
            # Exploitation: best action according to current policy
            with torch.no_grad():
                # Get Q-values for all actions
                q_values = self.policy_net(grid_state, extra_features)
                
                # Select best action
                return q_values.argmax().view(1, 1).to(self.device)
        
        # For non-training, just use the current policy
        with torch.no_grad():
            q_values = self.policy_net(grid_state, extra_features)
            return q_values.argmax().view(1, 1).to(self.device)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        try:
            transitions = self.memory.sample(self.batch_size)
            batch = list(zip(*transitions))
            
            # Get states
            states = batch[0]
            grid_states = torch.stack([s[0] for s in states]).to(self.device)
            extra_features = torch.stack([s[1] for s in states]).to(self.device)
            
            # Get actions and rewards
            action_batch = torch.cat(batch[1]).to(self.device)
            reward_batch = torch.FloatTensor(batch[2]).to(self.device)
            
            # Get next states
            next_states = batch[3]
            next_grid_states = torch.stack([s[0] for s in next_states]).to(self.device)
            next_extra_features = torch.stack([s[1] for s in next_states]).to(self.device)
            
            # Get done flags
            done_batch = torch.FloatTensor(batch[4]).to(self.device)
            
            # Debug print - only occasionally to reduce spam
            if self.current_step % 100 == 0:
                print(f"Batch shapes: grid={grid_states.shape}, extra={extra_features.shape}")
            
            # Add channel dimension if needed
            if grid_states.dim() == 3:
                grid_states = grid_states.unsqueeze(1)
            if next_grid_states.dim() == 3:
                next_grid_states = next_grid_states.unsqueeze(1)
            
            # Get Q values and actions
            current_q = self.policy_net(grid_states, extra_features).gather(1, action_batch)
            
            # Get target Q values - use done flag to zero out future rewards for terminal states
            # but don't apply any negative reinforcement
            with torch.no_grad():
                next_q = self.target_net(next_grid_states, next_extra_features).max(1)[0]
                # For terminal states, only use the immediate reward (no future discounting)
                # For non-terminal states, include the discounted future reward
                expected_q = (reward_batch + self.gamma * next_q * (1 - done_batch)).unsqueeze(1)
                
                # Ensure no negative Q-values (positive reinforcement only)
                expected_q = torch.max(expected_q, torch.zeros_like(expected_q))
            
            # Compute loss and optimize
            loss = nn.MSELoss()(current_q, expected_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network periodically
            if self.current_step % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
        except Exception as e:
            print(f"ERROR in optimize_model: {e}")
            if 'grid_states' in locals():
                print(f"Grid shape: {grid_states.shape}")
            if 'extra_features' in locals():
                print(f"Extra shape: {extra_features.shape}")
            # Don't raise the exception, just log it

    def save_model(self):
        # Save neural network state and training parameters
        network_path = 'train_network.pt'
        stats_path = 'train_network.json'
        
        try:
            # Save neural network state, optimizer state, and training parameters
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'steps_done': self.steps_done,
                'episodes_completed': self.episodes_completed,
                'device': str(self.device)
            }, network_path)
            
            # Save training statistics
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f)
            
            print("Training data saved successfully")
            
        except Exception as e:
            print(f"Error saving training data: {e}")

    def load_model(self):
        network_path = 'train_network.pt'
        stats_path = 'train_network.json'
        
        try:
            # Load neural network state if file exists
            if os.path.exists(network_path):
                checkpoint = torch.load(network_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.steps_done = checkpoint['steps_done']
                self.episodes_completed = checkpoint['episodes_completed']
                print("Neural network state loaded successfully")
            
            # Load training statistics if file exists
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
                print("Training statistics loaded successfully")
            
        except Exception as e:
            print(f"Error loading training data: {e}")
            # Initialize fresh training stats if loading fails
            self.training_stats = {
                'episodes': [],
                'scores': [],
                'avg_scores': [],
                'max_score': 0
            }

    def train_step(self, game):
        if self.episodes_completed >= self.training_episodes:
            game.training = False
            return

        try:
            # Ensure the game has a current block and next blocks
            if not game.current_block or not game.next_blocks:
                if not game.current_block:
                    print("No current block, generating one")
                    game.new_block()
                if not game.next_blocks:
                    print("No next blocks, generating them")
                    for _ in range(Config.PREVIEW_BLOCKS):
                        game.next_blocks.append(Block())
            
            # Get current state
            state = self.get_state(game)
            
            # Select and execute action
            action = self.select_action(state, training=True)
            reward = self.execute_action(game, action)
            
            # Get next state
            next_state = self.get_state(game)
            
            # Store transition - use 'done' flag but don't use it for negative reinforcement
            self.memory.push(state, action, reward, next_state, game.game_over)
            
            # Optimize model if enough samples
            if len(self.memory) >= self.batch_size:
                self.optimize_model()
            
            self.current_step += 1
            
            # Handle episode completion
            if game.game_over:
                self.episodes_completed += 1
                
                # Update stats
                self.training_stats['episodes'].append(self.episodes_completed)
                self.training_stats['scores'].append(game.score)
                avg_score = np.mean(self.training_stats['scores'][-100:])
                self.training_stats['avg_scores'].append(float(avg_score))
                
                if game.score > self.training_stats['max_score']:
                    self.training_stats['max_score'] = game.score
                    self.save_model()
                
                print(f"Episode {self.episodes_completed}/{self.training_episodes}: Score = {game.score}, Avg = {avg_score:.2f}")
                
                # Always reset the game and continue training if we haven't reached the target episodes
                if self.episodes_completed < self.training_episodes:
                    game.reset()
                    game.new_block()
                    for _ in range(Config.PREVIEW_BLOCKS):
                        game.next_blocks.append(Block())
                    game.game_over = False  # Ensure game_over is reset
                else:
                    # Only stop training if we've completed all episodes
                    game.training = False
                
        except Exception as e:
            print(f"Error in training step: {e}")
            import traceback
            traceback.print_exc()
            # Initialize state to prevent UnboundLocalError in exception handler
            state = None
            # Don't disable training mode on error, just try to recover
            if game.game_over:
                game.reset()
                game.new_block()
                for _ in range(Config.PREVIEW_BLOCKS):
                    game.next_blocks.append(Block())
                game.game_over = False

    def execute_action(self, game, action):
        if not game.current_block:
            print("No current block in execute_action")
            return 0
            
        # Store initial state
        initial_score = game.score
        initial_lines = game.lines_cleared_total
        original_x = game.current_block.x
        original_y = game.current_block.y
        original_shape = [row[:] for row in game.current_block.shape]
        
        try:
            # Execute the move
            action_value = action.item() if isinstance(action, torch.Tensor) else action
            rotation = action_value // Config.GRID_WIDTH
            position = action_value % Config.GRID_WIDTH
            
            # Execute rotations
            for _ in range(rotation):
                game.rotate_block()
            
            # Execute horizontal movement
            dx = position - game.current_block.x
            if dx < 0:
                for _ in range(abs(dx)):
                    game.move_block(-1, 0)
            else:
                for _ in range(dx):
                    game.move_block(1, 0)
            
            # Drop the piece
            drop_distance = 0
            while game.is_valid_position():
                game.current_block.y += 1
                drop_distance += 1
            game.current_block.y -= 1
            game.move_block(0, 1)  # This will trigger line clears and score updates
            
            # Calculate reward
            reward = 0
            
            # Reward for score increase (this includes line clears)
            score_increase = game.score - initial_score
            if score_increase > 0:
                reward += score_increase * 2  # Double the reward for score increases
            
            # Additional reward for lines cleared
            lines_cleared = game.lines_cleared_total - initial_lines
            if lines_cleared > 0:
                reward += lines_cleared * 150  # Increased base reward for clearing lines
                
                # Bonus for multiple lines
                if lines_cleared > 1:
                    reward += (3 ** lines_cleared) * 50  # Increased exponential bonus for multiple lines
            
            # Reward for dropping pieces (encourages playing)
            reward += drop_distance * 0.5  # Increased reward for dropping pieces
            
            # Reward for survival (instead of penalty for game over)
            if not game.game_over:
                reward += 1  # Small constant reward for staying alive
                
                # Additional reward for keeping the board low
                max_height = self.get_max_height(game.grid)
                if max_height < Config.GRID_HEIGHT / 2:
                    reward += 5  # Reward for keeping the board less than half full
            
            return max(0, reward)  # Ensure reward is never negative
            
        except Exception as e:
            print(f"Error in execute_action: {e}")
            import traceback
            traceback.print_exc()
            # Restore original position if there was an error
            if game.current_block:
                game.current_block.x = original_x
                game.current_block.y = original_y
                game.current_block.shape = original_shape
            return 0

    def count_potential_lines(self, game):
        """Count how many lines would be cleared if piece is placed"""
        if not game.current_block:
            return 0
            
        lines = 0
        try:
            # Check each row where the current piece is
            for y in range(game.current_block.y, min(game.current_block.y + len(game.current_block.shape), Config.GRID_HEIGHT)):
                # Count filled cells in row
                filled = sum(1 for x in range(Config.GRID_WIDTH) if 
                            (0 <= y < len(game.grid) and 0 <= x < len(game.grid[0]) and game.grid[y][x]) or 
                            (0 <= x - game.current_block.x < len(game.current_block.shape[0]) and 
                             0 <= y - game.current_block.y < len(game.current_block.shape) and 
                             game.current_block.shape[y - game.current_block.y][x - game.current_block.x]))
                if filled == Config.GRID_WIDTH:
                    lines += 1
        except Exception as e:
            print(f"Error in count_potential_lines: {e}")
            return 0
            
        return lines

    def get_max_height(self, grid):
        # Get the height of the highest block
        for y in range(len(grid)):
            if any(grid[y]):
                return len(grid) - y
        return 0

    def count_holes(self, grid):
        # Count empty cells that have a filled cell above them
        holes = 0
        for x in range(len(grid[0])):
            found_block = False
            for y in range(len(grid)):
                if grid[y][x]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def set_training_episodes(self, episodes):
        self.training_episodes = episodes
        self.episodes_completed = 0 