# Tetris with AI

A Python implementation of the classic Tetris game with manual play and heuristic-based AI capabilities.

**Developed by Arne Damvin**

## Features

- Classic Tetris gameplay with colorful graphics
- Heuristic-based AI using Pierre Dellacherie's algorithm
- Adjustable difficulty levels
- High score tracking

## Requirements

- Python 3.8+
- Pygame
- NumPy
- Matplotlib

See `requirements.txt` for complete dependency list.

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Playing the Game

To play Tetris manually:

```
python tetris.py
```

### Controls

- Left/Right Arrow: Move block horizontally
- Down Arrow: Accelerate block descent
- Up Arrow: Rotate block
- Space: Hard drop
- P: Pause game
- Esc: Quit game

### AI Mode

The game includes a heuristic-based AI using Pierre Dellacherie's algorithm that can play automatically:

- A: Toggle AI-assisted play

## Project Structure

- `tetris.py`: Main game implementation with pygame
- `tetris_ai.py`: Implementation of the heuristic-based AI using Pierre Dellacherie's algorithm
- `block.py`: Block definitions and operations
- `colors.py`: Color definitions
- `config.py`: Game configuration settings
- `highscore.txt`: File storing the highest score achieved

## Customization

Game settings can be modified in `config.py`, including:
- Window dimensions
- Grid size and dimensions
- FPS limit

## License

Copyright (C) 2024 Arne Damvin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

