import pygame
from colors import Colors
from block import Block
from config import Config

class TetrisAI:
    def __init__(self):
        # Pierre Dellacherie's algorithm weights
        self.LANDING_HEIGHT_WEIGHT = -4.500158825082766
        self.ROWS_ELIMINATED_WEIGHT = 3.4181268101392694
        self.ROW_TRANSITIONS_WEIGHT = -3.2178882868487753
        self.COLUMN_TRANSITIONS_WEIGHT = -9.348695305445199
        self.HOLES_WEIGHT = -7.899265427351652
        self.WELL_SUMS_WEIGHT = -3.3855972247263626
        
        # Cache for column heights
        self.heights_cache = {}

    def get_best_move(self, game, current_block, next_blocks):
        best_score = float('-inf')
        best_move = None
        original_x = current_block.x
        original_y = current_block.y
        original_shape = [row[:] for row in current_block.shape]
        
        try:
            # Pre-calculate grid string for cache key
            grid_key = str(game.grid)
            if grid_key not in self.heights_cache:
                self.heights_cache[grid_key] = self._calculate_column_heights(game.grid)
            
            # Try all possible rotations and positions for current block
            for rotation in range(4):
                # Reset position and apply rotation
                current_block.shape = [row[:] for row in original_shape]
                current_block.x = original_x
                current_block.y = original_y
                
                for _ in range(rotation):
                    current_block.rotate()
                
                if not game.is_valid_position():
                    continue
                
                # Optimize x-position search using block width
                block_width = len(current_block.shape[0])
                min_x = 0
                max_x = Config.GRID_WIDTH - block_width
                
                for x in range(min_x, max_x + 1):
                    current_block.x = x
                    if not game.is_valid_position():
                        continue
                    
                    # Find landing position using column heights
                    landing_y = self._find_landing_height(game.grid, current_block, self.heights_cache[grid_key])
                    if landing_y is None:
                        continue
                    
                    current_block.y = landing_y
                    
                    # Create a copy of the grid with the current piece placed
                    test_grid = [row[:] for row in game.grid]
                    self._place_piece(test_grid, current_block)
                    
                    # Evaluate current position
                    current_score = self._evaluate_position(test_grid, landing_y, current_block)
                    
                    # Look ahead - consider all available preview blocks
                    next_score = 0
                    look_ahead_weight = 0.4
                    
                    # Use a decreasing weight for each next block (further blocks have less influence)
                    if next_blocks and len(next_blocks) > 0:
                        # Calculate how many blocks to consider (limited by available blocks)
                        blocks_to_consider = min(Config.PREVIEW_BLOCKS, len(next_blocks))
                        
                        for i in range(blocks_to_consider):
                            # Weight decreases for each future block
                            block_weight = look_ahead_weight / (i + 1)
                            
                            # Evaluate this next block
                            block_score = self._quick_evaluate_next_move(test_grid, next_blocks[i])
                            next_score += block_weight * block_score
                    
                    # Combine scores
                    score = current_score + next_score
                    
                    if score > best_score:
                        best_score = score
                        best_move = (rotation, x - original_x)
        except Exception as e:
            print(f"Error in get_best_move: {e}")
        finally:
            # Always restore block to original state
            current_block.shape = original_shape
            current_block.x = original_x
            current_block.y = original_y
            
            # Clear cache if it gets too large
            if len(self.heights_cache) > 1000:
                self.heights_cache.clear()
        
        return best_move

    def _calculate_column_heights(self, grid):
        """Calculate and cache the height of each column"""
        heights = [0] * len(grid[0])
        for x in range(len(grid[0])):
            for y in range(len(grid)):
                if grid[y][x]:
                    heights[x] = len(grid) - y
                    break
        return heights

    def _find_landing_height(self, grid, block, column_heights):
        """Find landing height efficiently using column heights"""
        max_height = 0
        block_width = len(block.shape[0])
        block_height = len(block.shape)
        
        # Find the maximum height in the affected columns
        for x in range(block_width):
            if block.x + x >= 0 and block.x + x < len(grid[0]):
                max_height = max(max_height, column_heights[block.x + x])
        
        # Start checking from the highest point
        y = max(0, len(grid) - max_height - block_height)
        
        # Save original y position to restore later
        original_y = block.y
        
        # Check positions from top to bottom
        while y < len(grid):
            block.y = y
            if not self._is_valid_position(grid, block):
                # Restore original position before returning
                block.y = original_y
                return y - 1 if y > 0 else None
            y += 1
        
        # Restore original position before returning
        block.y = original_y
        return len(grid) - 1

    def _quick_evaluate_next_move(self, grid, next_block):
        """Faster evaluation of next move without recursion"""
        best_score = float('-inf')
        original_shape = [row[:] for row in next_block.shape]
        original_x = next_block.x
        original_y = next_block.y
        
        try:
            for rotation in range(4):
                next_block.shape = [row[:] for row in original_shape]
                for _ in range(rotation):
                    next_block.rotate()
                
                block_width = len(next_block.shape[0])
                for x in range(Config.GRID_WIDTH - block_width + 1):
                    next_block.x = x
                    next_block.y = 0
                    
                    # Quick height check
                    landing_y = self._find_quick_landing_height(grid, next_block)
                    if landing_y is not None:
                        next_block.y = landing_y
                        test_grid = [row[:] for row in grid]
                        self._place_piece(test_grid, next_block)
                        score = self._evaluate_position(test_grid, landing_y, next_block)
                        best_score = max(best_score, score)
        except Exception as e:
            print(f"Error in quick evaluate next move: {e}")
        finally:
            # Always restore original state
            next_block.shape = original_shape
            next_block.x = original_x
            next_block.y = original_y
            
        return best_score

    def _find_quick_landing_height(self, grid, block):
        """Simplified landing height check"""
        original_y = block.y
        
        for y in range(len(grid) - len(block.shape), -1, -1):
            block.y = y
            if self._is_valid_position(grid, block):
                # Found valid position, restore original y before returning
                result = y
                block.y = original_y
                return result
                
        # No valid position found, restore original y before returning
        block.y = original_y
        return None

    def _place_piece(self, grid, block):
        """Place the current piece in the test grid"""
        for y, row in enumerate(block.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_y = block.y + y
                    grid_x = block.x + x
                    if 0 <= grid_y < len(grid) and 0 <= grid_x < len(grid[0]):
                        grid[grid_y][grid_x] = 1

    def _evaluate_position(self, grid, landing_y, block):
        """Evaluate a potential move using Dellacherie's algorithm"""
        try:
            landing_height = Config.GRID_HEIGHT - landing_y if landing_y is not None else Config.GRID_HEIGHT
            rows_eliminated = self._count_eliminated_rows(grid)
            row_transitions = self._count_row_transitions(grid)
            column_transitions = self._count_column_transitions(grid)
            holes = self._count_holes(grid)
            well_sums = self._calculate_well_sums(grid)
            
            return float(
                self.LANDING_HEIGHT_WEIGHT * landing_height +
                self.ROWS_ELIMINATED_WEIGHT * rows_eliminated +
                self.ROW_TRANSITIONS_WEIGHT * row_transitions +
                self.COLUMN_TRANSITIONS_WEIGHT * column_transitions +
                self.HOLES_WEIGHT * holes +
                self.WELL_SUMS_WEIGHT * well_sums
            )
        except Exception as e:
            print(f"Error in position evaluation: {e}")
            return float('-inf')

    def _count_eliminated_rows(self, grid):
        """Count number of complete lines"""
        return sum(1 for row in grid if all(cell for cell in row))

    def _count_row_transitions(self, grid):
        """Count number of row transitions"""
        transitions = 0
        for row in grid:
            # Count transitions between filled/empty cells and edges
            prev = 1  # Consider edges as filled
            for cell in row:
                if cell != prev:
                    transitions += 1
                prev = cell
            if prev == 0:  # Transition to right edge
                transitions += 1
        return transitions

    def _count_column_transitions(self, grid):
        """Count number of column transitions"""
        transitions = 0
        for x in range(len(grid[0])):
            # Count transitions between filled/empty cells and edges
            prev = 1  # Consider edges as filled
            for y in range(len(grid)):
                cell = grid[y][x]
                if cell != prev:
                    transitions += 1
                prev = cell
            if prev == 0:  # Transition to bottom edge
                transitions += 1
        return transitions

    def _count_holes(self, grid):
        """Count empty cells with filled cells above them"""
        holes = 0
        for x in range(len(grid[0])):
            block_found = False
            for y in range(len(grid)):
                if grid[y][x]:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def _calculate_well_sums(self, grid):
        """Calculate well sums"""
        try:
            well_sums = 0
            width = len(grid[0])
            height = len(grid)
            
            # Check each column
            for x in range(width):
                for y in range(height):
                    # Check if cell is empty and has blocks on both sides (or edge)
                    is_well = not grid[y][x]
                    if is_well:
                        if x == 0:  # Left edge
                            is_well = x + 1 < width and grid[y][x + 1]
                        elif x == width - 1:  # Right edge
                            is_well = grid[y][x - 1]
                        else:  # Middle columns
                            is_well = grid[y][x - 1] and grid[y][x + 1]
                    
                    if is_well:
                        # Count well depth
                        depth = 0
                        for well_y in range(y, height):
                            if not grid[well_y][x]:
                                depth += 1
                            else:
                                break
                        well_sums += depth
            
            return well_sums
        except Exception as e:
            print(f"Error in well sums calculation: {e}")
            return 0

    def _is_valid_position(self, grid, block):
        """Check if the block's position is valid in the given grid"""
        for y, row in enumerate(block.shape):
            for x, cell in enumerate(row):
                if cell:
                    pos_x = block.x + x
                    pos_y = block.y + y
                    if (pos_x < 0 or pos_x >= Config.GRID_WIDTH or
                        pos_y >= Config.GRID_HEIGHT or pos_y < 0 or
                        (pos_y >= 0 and grid[pos_y][pos_x])):
                        return False
        return True 