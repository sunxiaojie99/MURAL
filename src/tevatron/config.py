from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Optional, Union

from regex import F

__ALL__ = ["get_cfg"]
KEY = "config"


def get_filename(path: str) -> str:
    filename = path.split("/")[-1].split(".")[0]
    return filename


def get_cfg(path: Optional[str] = None) -> DictConfig:
    if path is not None:
        cfg = OmegaConf.load(path)
        cfg = OmegaConf.merge(_C, cfg)
        cfg.experiment.name = get_filename(path)
    else:
        cfg = _C.copy()
        cfg.experiment.name = "NA"
    return cfg  # type: ignore


_C = OmegaConf.create()
_C.experiment = OmegaConf.create()
_C.experiment.name = ""
