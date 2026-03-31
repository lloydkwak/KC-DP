import os
import sys
import torch
from omegaconf import OmegaConf

# ============================================================================
# Register the Absolute Path Resolver before calling the baseline script
# ============================================================================
def load_stats(path: str, key: str):
    """
    Loads statistics dynamically. Immune to Hydra's outputs/ folder routing.
    """
    if not os.path.isabs(path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, path)

    if os.path.exists(path):
        try:
            stats = torch.load(path)
            return stats[key]
        except Exception as e:
            pass
    return None

try:
    OmegaConf.register_new_resolver("load_stats", load_stats, replace=True)
except ValueError:
    pass

# ============================================================================
# Execute the upstream diffusion_policy training loop
# ============================================================================
# Ensure PYTHONPATH points to the root of the diffusion_policy repository.
from diffusion_policy.scripts.train import main

if __name__ == "__main__":
    main()