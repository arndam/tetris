import torch

class Config:
    # Game settings
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    GRID_SIZE = 30
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    GAME_AREA_LEFT = 400
    GAME_AREA_TOP = 50
    FPS = 60
    
    # Preview settings
    PREVIEW_BLOCKS = 2  # Number of blocks to show in preview and use for AI
    
    # Training settings
    USE_CUDA = True
    DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")
    TRAINING_EPISODES = 1000 