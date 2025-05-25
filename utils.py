# utils.py  : Utility functions for reproducibility and scheduling.
#
# Author       : Casper Bröcheler <casper.jxb@gmail.com>
# GitHub       : https://github.com/casperbroch
# Affiliation  : Maastricht University


import torch
import numpy as np
import random

# Seed Python, NumPy, and PyTorch (incl. CUDA) for reproducibility
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Return a SB3-style linear schedule lambda
def linear_schedule(initial_value: float):
    return lambda progress_remaining: progress_remaining * initial_value
