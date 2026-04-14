import os
from typing import Optional

from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import (
    RobomimicReplayLowdimDataset,
)


class TaskSpaceLowdimDataset(RobomimicReplayLowdimDataset):
    """
    RemoteBot task-space dataset wrapper.

    - Removes KC-DP specific k(q) augmentation path.
    - Keeps robust dataset path resolution for host / docker layouts.
    - Reads low-dim observation + action as-is from Robomimic HDF5.
    """

    def __init__(self, *args, dataset_path: Optional[str] = None, **kwargs):
        if args:
            resolved = self._resolve_dataset_path(str(args[0]))
            args = (resolved, *args[1:])
        else:
            if dataset_path is None:
                dataset_path = kwargs.get("dataset_path", None)
            if dataset_path is None:
                raise ValueError("dataset_path must be provided.")
            kwargs["dataset_path"] = self._resolve_dataset_path(str(dataset_path))

        kwargs.pop("zarr_path", None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _resolve_dataset_path(dataset_path: str) -> str:
        if os.path.exists(dataset_path):
            return dataset_path

        if dataset_path.startswith("data/robomimic/"):
            cand = dataset_path.replace(
                "data/robomimic/",
                "third_party/diffusion_policy/data/robomimic/",
                1,
            )
            if os.path.exists(cand):
                return cand

        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        cand2 = os.path.join(project_root, dataset_path)
        if os.path.exists(cand2):
            return cand2

        if "data/robomimic/" in dataset_path:
            rel = dataset_path.split("data/robomimic/", 1)[1]
            cand3 = os.path.join(
                project_root,
                "third_party",
                "diffusion_policy",
                "data",
                "robomimic",
                rel,
            )
            if os.path.exists(cand3):
                return cand3

        raise FileNotFoundError(
            f"Dataset not found: '{dataset_path}'."
        )
