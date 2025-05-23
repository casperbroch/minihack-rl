import torch
import numpy as np
import random

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def linear_schedule(initial_value: float):
    """SB3-style linear lr/clip schedule."""
    return lambda progress_remaining: progress_remaining * initial_value
