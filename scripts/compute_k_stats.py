import os
import torch
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf

# ============================================================================
# Absolute Path Resolver for Hydra Configs
# ============================================================================
def load_stats(path: str, key: str):
    """
    Safely loads normalization statistics. Uses absolute paths based on the 
    project root to prevent Hydra's dynamic working directory issues.
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

@hydra.main(config_path="../configs", config_name="train_kc_dp", version_base="1.2")
def main(cfg):
    """
    Iterates over the dataset to compute global mean and std for k(q) features.
    """
    print("Starting k(q) statistics computation with kinematic augmentation...")
    
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    num_samples = len(dataset)
    print(f"Total trajectory windows to process: {num_samples}")

    all_k_q = []
    
    for i in tqdm(range(num_samples), desc="Extracting k(q)"):
        # Periodically clear the cache to ensure morphological diversity
        if i % 100 == 0:
            dataset.clear_epoch_cache()
            
        item = dataset[i]
        
        # Slice only the raw k_q dimensions appended at the end of the tensor
        k_q = item['obs'][..., -42:] 
        all_k_q.append(k_q.reshape(-1, 42))

    all_k_q_tensor = torch.cat(all_k_q, dim=0)
    
    k_mean = all_k_q_tensor.mean(dim=0).numpy().tolist()
    k_std = all_k_q_tensor.std(dim=0).numpy()
    
    # Clip standard deviation to prevent future division-by-zero
    k_std = np.clip(k_std, a_min=1e-6, a_max=None).tolist()
    
    # Anchor the output path to the absolute project root
    output_path = "data/k_q_stats.pt"
    if not os.path.isabs(output_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(project_root, output_path)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    stats_dict = {'mean': k_mean, 'std': k_std}
    torch.save(stats_dict, output_path)
    
    print(f"\nStatistics computation completed successfully!")
    print(f"Saved normalization constants to: {output_path}")

if __name__ == "__main__":
    main()